import torch
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
import numpy as np
import os
import h5py

from .utils import crop_image_to_bb, get_refcoco_data, compute_position_features, pad_img_to_max, xywh_to_xyxy


class RefCocoDataset(Dataset):

    def __init__(self,
                 data,
                 img_root,
                 max_length,
                 transform,
                 tokenizer,
                 prefix_length,
                 add_eos=True,
                 normalize_prefix=False,
                 return_unique=False,
                 return_global_context=False,
                 return_location_features=False,
                 return_scene_features=False,
                 scene_summary_ids=None,
                 scene_summary_features=None,
                 return_tensor=True,
                 return_original_image=False
                 ):
        super().__init__()

        self.img_root = img_root
        self.transform = transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]
        
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix        

        # flags for input composition
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_tensor = return_tensor
        self.return_original_image = return_original_image
        self.return_scene_features = return_scene_features
        self.scene_summary_ids = scene_summary_ids
        self.scene_summary_features = scene_summary_features

        if return_unique:
            # filter for unique ids
            self.annot_select = []
            stored_ids = []
            for a in self.annot:
                if a[0] not in stored_ids:
                    self.annot_select.append(a)
                    stored_ids.append(a[0])
        else:
            self.annot_select = self.annot

        self.tokenizer = tokenizer
        self.max_length = max_length + 1
        self.add_eos = add_eos
        if self.add_eos:
            self.eos_token = self.tokenizer.eos_token
            eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
            self.eos_suffix = [eos_id]
        else:
            self.eos_suffix = []
            
        if isinstance(self.transform, dict):
            assert set(self.transform.keys()) == {'target', 'context'}
            self.target_transform = self.transform['target']
            self.context_transform = self.transform['context']
        else:
            self.target_transform = self.context_transform = self.transform
        

    def pad_tokens(self, tokens):
        padding = self.max_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_length]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return 'COCO_train2014_' + val + '.jpg'

    def __len__(self):
        return len(self.annot_select)
    
    def get_imgs_from_ann_id(self, ann_id):
        annot_dict = dict([(a[0], a[1:]) for a in self.annot_select])
        image_file, caption, bb = annot_dict[ann_id]

        image_filepath = os.path.join(self.img_root, 'train2014', image_file)
        assert os.path.isfile(image_filepath)
        image = Image.open(image_filepath)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        target_image, _, context_image, _ = crop_image_to_bb( # type: ignore
            image, bb, return_context=True)
        
        return image, target_image, context_image, caption
    
    def get_bbox_from_ann_id(self, ann_id):
        annot_dict = dict([(a[0], a[1:]) for a in self.annot_select])
        _, _, bb = annot_dict[ann_id]
        return bb
    
    def get_annotated_image(self, ann_id, return_caption=False, bbox_color='blue', width=3):
        full_image, _, _, caption = self.get_imgs_from_ann_id(ann_id)
        bbox = self.get_bbox_from_ann_id(ann_id)
        bbox_xyxy = xywh_to_xyxy(bbox)
        
        draw = ImageDraw.Draw(full_image)
        draw.rectangle(bbox_xyxy, outline=bbox_color, width=width)
        
        return full_image if not return_caption else (full_image, caption)


    def __getitem__(self, idx):
        ann_id, image_file, caption, bb = self.annot_select[idx]
        image_filepath = os.path.join(self.img_root, 'train2014', image_file)
        assert os.path.isfile(image_filepath)

        image = Image.open(image_filepath)

        # CAPTION
        
        caption = torch.tensor(
            self.tokenizer.encode(caption) + self.eos_suffix, 
            dtype=torch.int64
        )
        caption, cap_mask = self.pad_tokens(caption)

        # IMAGE

        # convert if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # crop to bounding box
        target_image, _, context_image, _ = crop_image_to_bb( # type: ignore
            image, bb, return_context=True)

        # target bb
        target_image = pad_img_to_max(target_image)
        target_image = self.target_transform(target_image)

        # base case: only target representation as encoder input
        encoder_input = [target_image]

        if self.return_global_context:
            # add global context
            context_image = pad_img_to_max(context_image)
            context_image = self.context_transform(context_image)
            
            encoder_input += [context_image]
            
        if self.return_scene_features: 
            # add scene summaries
            selection_mask = self.scene_summary_ids == ann_id
            scene_summary = torch.from_numpy(
                self.scene_summary_features[selection_mask]).squeeze()
            encoder_input.append(scene_summary)
            
        if self.return_location_features:
            # add location features
            position_feature = compute_position_features(image, bb)
            encoder_input += [position_feature]

        if self.return_original_image: 
            return ann_id, *encoder_input, caption, cap_mask, [image]

        return ann_id, *encoder_input, caption, cap_mask


def build_dataset(transform,
                  tokenizer,
                  ann_dir,
                  img_dir,
                  prefix_length=10,
                  verbose=True,
                  max_length=20,
                  mode='training',
                  use_global_features=True,
                  use_location_features=True,
                  scenesum_dir=None,
                  use_scene_summaries=False,
                  return_unique=False, 
                  return_tensor=True,
                  return_original_image=False):

    assert mode in ['training', 'train', 'validation', 'val', 'testa', 'testb', 'test']

    full_data, ids = get_refcoco_data(ann_dir)

    # select data partition

    if mode.lower() in ['training', 'train']:
        partition = 'train'
    elif mode.lower() in ['validation', 'val']:
        partition = 'val'
    elif mode.lower() == 'testa':  # refcoco / refcoco+
        partition = 'testA'
    elif mode.lower() == 'testb':  # refcoco / refcoco+
        partition = 'testB'
    elif mode.lower() == 'test':  # refcocog
        partition = 'test'
    else:
        raise NotImplementedError(f"{mode} not supported")

    data = full_data.loc[ids['caption_ids'][partition]]

    # handle scene summaries if set in config
    if use_scene_summaries:
        scenesum_filepath = os.path.join(
            scenesum_dir, "scene_summaries", f"scene_summaries_annotated_{partition}.h5"
        )
        print(f"read scene summaries from {scenesum_filepath}")
        with h5py.File(scenesum_filepath, "r") as f:
            scenesum_ann_ids = f["ann_ids"][:].squeeze(1)
            scenesum_feats = f["context_feats"][:]
    else:
        scenesum_ann_ids = scenesum_feats = None

    # build dataset
    dataset = RefCocoDataset(
        data=data.to_dict(orient='records'),
        img_root=img_dir,
        max_length=max_length,
        transform=transform,
        tokenizer=tokenizer,
        prefix_length=prefix_length,
        return_unique=return_unique,
        return_global_context=use_global_features,
        return_location_features=use_location_features, 
        return_scene_features=use_scene_summaries,
        scene_summary_ids=scenesum_ann_ids,
        scene_summary_features=scenesum_feats,
        return_tensor=return_tensor,
        return_original_image=return_original_image
        )

    if verbose:
        print(f'Initialize {dataset.__class__.__name__} with mode: {partition}', 
            '\ntransformation:', transform, 
            f'\nentries: {len(dataset)}',
            '\nreturn unique:', return_unique, '\n')  

    return dataset
