import os
import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision.transforms import Compose
import sys
from transform import PrepareForNet, Resize, NormalizeImage, Crop

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class CityscapesDepthLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(518, 518),
        depth_size=(518, 518),
        additional_transforms=None
    ):
        self.mode = 'train'
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}
        self.depth_size = depth_size if isinstance(depth_size, tuple) else (depth_size, depth_size)
        net_h, net_w = img_size
        self.transform_image = Compose([
            Resize(
                width=net_w, 
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]+ ([Crop(img_size[0])] if self.mode == 'train' else []))
        self.additional_transforms = additional_transforms

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.depth_base = os.path.join(self.root, "crestereo_depth", self.split)
        
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        # Prints number of images found
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
    

        name = '_'.join(os.path.basename(img_path).split('_')[:3]) + '_crestereo_depth.npy'
        depth_path = os.path.join(
            self.depth_base,
            img_path.split(os.sep)[-2],
            name
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        if self.is_transform:
            img, depth = self.transform(img, depth)

        valid_mask = depth <= 80
        return img, depth, valid_mask

    def transform(self, img, depth):
        if self.additional_transforms != None:
            transformed = self.additional_transforms(image=img, mask=depth)
            img = transformed['image']
            depth = transformed['mask']

        img = img / 255.0
        sample = self.transform_image({'image': img, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        img = sample['image']
        depth = sample['depth']
        
        return img, depth