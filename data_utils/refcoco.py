import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import os

from .utils import crop_image_to_bb, get_refcoco_data, compute_position_features, pad_img_to_max


class RefCocoDataset(Dataset):

    def __init__(self,
                 data,
                 root,
                 max_length,
                 transform,
                 tokenizer,
                 prefix_length,
                 add_eos=True,
                 normalize_prefix=False,
                 return_unique=False,
                 return_global_context=False,
                 return_location_features=False,
                 return_tensor=True,
                 ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]
        
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix        

        # flags for input composition
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_tensor = return_tensor

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

        image_filepath = os.path.join(self.root, 'train2014', image_file)
        assert os.path.isfile(image_filepath)
        image = Image.open(image_filepath)

        target_image, _, context_image, _ = crop_image_to_bb(
            image, bb, return_context=True)
        
        return target_image, context_image, caption


    def __getitem__(self, idx):
        ann_id, image_file, caption, bb = self.annot_select[idx]
        image_filepath = os.path.join(self.root, 'train2014', image_file)
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
        target_image, target_mask, context_image, context_mask = crop_image_to_bb(
            image, bb, return_context=True)

        # target bb
        target_image = pad_img_to_max(target_image)
        target_image = self.transform(target_image)

        encoder_input = [target_image]

        if self.return_global_context:
            # add global context
            context_image = pad_img_to_max(context_image)
            context_image = self.transform(context_image)
            
            encoder_input += [context_image]
            
        if self.return_location_features:
            # add location features
            position_feature = compute_position_features(image, bb)
            encoder_input += [position_feature]

        return ann_id, *encoder_input, caption, cap_mask


def build_dataset(transform,
                  tokenizer,
                  ref_dir,
                  coco_dir,
                  verbose=True,
                  max_length=20,
                  mode='training',
                  use_global_features=True,
                  use_location_features=True,
                  return_unique=False, 
                  return_tensor=True):

    assert mode in ['training', 'train', 'validation', 'val', 'testa', 'testb', 'test']

    full_data, ids = get_refcoco_data(ref_dir)

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
       
    if verbose:
        print(f'Initialize Dataset with mode: {partition}', 
            '\ntransformation:', transform, 
            f'\nentries: {len(data)}',
            '\nreturn unique:', return_unique, '\n')

    # build dataset
    dataset = RefCocoDataset(data=data.to_dict(orient='records'),
                             root=coco_dir,
                             max_length=max_length,
                             transform=transform,
                             tokenizer=tokenizer,
                             prefix_length=10,
                             return_unique=return_unique,
                             return_global_context=use_global_features,
                             return_location_features=use_location_features, 
                             return_tensor=return_tensor)
    return dataset
