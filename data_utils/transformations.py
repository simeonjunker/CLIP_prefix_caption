import numpy as np
from torchvision.transforms import ToTensor, ToPILImage, Pad, Compose
import torch
from random import random
from copy import deepcopy


class SquarePad:
    def __call__(self, image, pad_value=0):
        w, h = image.size
        max_wh = max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        pad_fn = Pad(padding=padding, fill=pad_value, padding_mode='constant')
        return pad_fn(image)


class AddNoise:
	def __call__(self, input_image, noise_factor=0.5):
		inputs = ToTensor()(input_image)
		noise = inputs + torch.rand_like(inputs) * noise_factor
		noise = torch.clip (noise,0,1.)
		output_image = ToPILImage()
		image = output_image(noise)
		return image


class CoverWithNoise:
    
    def __init__(self, noise_coverage=0.5):
        self.noise_coverage = noise_coverage
        
    def __str__(self):
        return f"CoverWithNoise(noise_coverage={self.noise_coverage})"
        
    def __call__(self, image):
        image_tensor = ToTensor()(image)
        # create 1D selection mask
        _, h, w = image_tensor.shape
        mask = np.zeros((h, w), dtype=bool).flatten()  # [hw]

        # sample from flattened index & set mask to True
        idx = np.indices(mask.shape).flatten()
        idx_sample = np.random.choice(idx, replace=False, size=round(mask.size * self.noise_coverage))
        mask[idx_sample] = True
        # reshape to 2D image shape
        mask = torch.from_numpy(mask.reshape((h, w)))  # [h, w]

        # mask image with noise
        noise_tensor = torch.rand_like(image_tensor)
        image_tensor[:, mask] = noise_tensor[:, mask]

        return ToPILImage()(image_tensor)
    

class SometimesCoverWithNoise:
    
    def __init__(self, noise_coverage=0.5, chance=0.5):
        self.noise_coverage = noise_coverage
        self.chance = chance
        
    def __str__(self):
        return f"SometimesCoverWithNoise(noise_coverage={self.noise_coverage}, chance={self.chance})"
        
    def __call__(self, image):
        
        if random() > 1 - self.chance:
                    
            image_tensor = ToTensor()(image)
            # create 1D selection mask
            _, h, w = image_tensor.shape
            mask = np.zeros((h, w), dtype=bool).flatten()  # [hw]

            # sample from flattened index & set mask to True
            idx = np.indices(mask.shape).flatten()
            idx_sample = np.random.choice(idx, replace=False, size=round(mask.size * self.noise_coverage))
            mask[idx_sample] = True
            # reshape to 2D image shape
            mask = torch.from_numpy(mask.reshape((h, w)))  # [h, w]

            # mask image with noise
            noise_tensor = torch.rand_like(image_tensor)
            image_tensor[:, mask] = noise_tensor[:, mask]

            image = ToPILImage()(image_tensor)

        return image

    
    
def update_transforms(transforms, pad_transform=None, noise_transform=None):
    
    transforms_list = deepcopy(transforms).transforms
    
    if pad_transform is not None:
        # add as initial transformation
        transforms_list.insert(0, pad_transform)
        
    if noise_transform is not None:
        # add before conversion to tensor
        to_tensor_idx = [
            i for i, t in enumerate(transforms_list) 
            if t.__class__ == ToTensor
        ][0]
        
        transforms_list.insert(to_tensor_idx, noise_transform)
    
    return Compose(transforms_list)