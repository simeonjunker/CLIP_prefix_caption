import torch
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
import numpy as np
import os

from .utils import crop_image_to_bb, get_paco_df, compute_position_features, pad_img_to_max, xywh_to_xyxy


class PACODataset(Dataset):

    def __init__(self,
                 data,
                 img_root,
                 max_length,
                 transform,
                 tokenizer,
                 prefix_length,
                 add_eos=True,
                 normalize_prefix=False,
                 return_global_context=False,
                 return_location_features=False,
                 return_tensor=True,
                 return_original_image=False
                 ):
        super().__init__()

        self.img_root = img_root
        self.transform = transform
        self.annot = [(entry['index'], entry['file_name'],
                       entry['name'], entry['bbox']) for entry in data]
        
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix        

        # flags for input composition
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_tensor = return_tensor
        self.return_original_image = return_original_image

        # no return_unique in PACODataset
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

    def __len__(self):
        return len(self.annot_select)
    
    def get_imgs_from_ann_id(self, ann_id):
        annot_dict = dict([(a[0], a[1:]) for a in self.annot_select])
        image_file, caption, bb = annot_dict[ann_id]

        image_filepath = os.path.join(self.img_root, image_file)
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
        image_filepath = os.path.join(self.img_root, image_file)
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

        encoder_input = [target_image]

        if self.return_global_context:
            # add global context
            context_image = pad_img_to_max(context_image)
            context_image = self.context_transform(context_image)
            
            encoder_input += [context_image]
            
        if self.return_location_features:
            # add location features
            position_feature = compute_position_features(image, bb)
            encoder_input += [position_feature]

        if self.return_original_image: 
            return ann_id, *encoder_input, caption, cap_mask, [image]

        return ann_id, *encoder_input, caption, cap_mask


def process_part_names(df, only='part'):
    
    parts_df = df.loc[df.supercategory == 'PART']
    
    if only == 'part':
        parts_df['name'] = parts_df['name'].map(lambda x: x.split(':')[-1]) # keep part after ':'
    elif only == 'full':
        parts_df['name'] = parts_df['name'].map(lambda x: x.split(':')[0])  # keep part before ':'
    else:
        raise NotImplementedError(f"'only' value has to by 'part' or 'full', but is {only}")
    
    df.loc[parts_df.index] = parts_df
    
    return df


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
                  return_unique=False, 
                  return_tensor=True,
                  return_original_image=False,
                  parts_only_part=False,
                  parts_only_full=False):

    assert mode in ['training', 'train', 'validation', 'val', 'test'], f"{mode} not supported"
    if mode == 'training':
        mode = 'train'
    elif mode == 'validation': 
        mode = 'val'
    elif mode == 'test':
        mode = 'test_dev'
        
    paco_ann_file = f'paco_ego4d_v1_{mode}.json'
    paco_ann_path = os.path.join(ann_dir, paco_ann_file)

    data = get_paco_df(paco_ann_path).reset_index()
    
    if parts_only_part:
        data = process_part_names(data, only='part')
    elif parts_only_full:
        data = process_part_names(data, only='full')
       
    # build dataset
    dataset = PACODataset(
        data=data.to_dict(orient='records'),
        img_root=img_dir,
        max_length=max_length,
        transform=transform,
        tokenizer=tokenizer,
        prefix_length=prefix_length,
        return_global_context=use_global_features,
        return_location_features=use_location_features, 
        return_tensor=return_tensor,
        return_original_image=return_original_image
    )
    
    if verbose:
        print(
            f'Initialize {dataset.__class__.__name__} with mode: {mode}', 
            '\ntransformation:', transform, 
            f'\nentries: {len(dataset)}',
            '\nreturn unique (without function in PACO):', return_unique,
            '\nreturn only part names (for parts):', parts_only_part,
            '\nreturn only full names (for parts):', parts_only_full, 
            '\n'
        )
        
    return dataset
