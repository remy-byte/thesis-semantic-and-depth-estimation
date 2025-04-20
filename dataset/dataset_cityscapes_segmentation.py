import torch
import os
import numpy as np
import scipy.misc as m
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as skm
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt  
import torch.nn.functional as F
import time
import cv2
from torchvision import transforms
from .transform import PrepareForNet, Resize, NormalizeImage
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class cityscapesLoader(data.Dataset):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        # which data split to use
        split="train",
        # transform function activation
        is_transform=True,
        # image_size to use in transform function
        img_size=(518,518),
        lbl_size=(1024, 2048),
        additional_transforms=None
    ):
        self.mode = 'train'
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}
        self.lbl_size = lbl_size if isinstance(lbl_size, tuple) else (lbl_size, lbl_size)
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
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
]

        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall", 
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        # self.class_names_v2 = [
        #     "unlabelled",
        #     "road",
        #     "sidewalk",
        #     "building",
        #     "traffic_light",
        #     "traffic_sign",
        #     "vegetation",
        #     "person",
        #     "car",
        #     "bus",
        #     "motorcycle",
        #     "bicycle"
        # ]

        # self.void_classes = [1, 2, 3, 4, 5, 6, 9, 10, 12, 13 ,14, 15, 16, 17, 23, 22,25, 27, 18, 29, 30, 31, -1]

        # self.valid_classes = [0, 7, 8, 11, 19, 20, 21, 24, 26, 28, 32, 33]


        
        # for void_classes; useful for loss function
        self.ignore_index = 255
        
        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        # prints number of images found
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
    
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        # print(img_path, lbl_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(lbl_path, -1)
        lbl = self.encode_segmap(lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # lbl = lbl.astype(float)
        # img = img.astype(np.float64)
    # NHWC -> NCHW
        # img = img.transpose(2, 0, 1)
        if self.additional_transforms != None:
            transformed = self.additional_transforms(image = img, mask = lbl)
            img = transformed['image']
            lbl = transformed['mask']
        classes = np.unique(lbl)

        img = img /255.
        sample = self.transform_image({'image': img, 'semseg_mask': lbl})
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['semseg_mask'] = torch.from_numpy(sample['semseg_mask']).long()

        img = sample['image']
        lbl = sample['semseg_mask']
        # lbl = cv2.resize(lbl, (self.lbl_size[1], self.lbl_size[0]), interpolation=cv2.INTER_NEAREST)
        # lbl = lbl.astype(int)

        try:
            if not np.all(classes == np.unique(lbl)):
                print('classes', classes)
                print('mask', np.unique(lbl))
                print("WARN: resizing labels yielded fewer classes")
        except:
            print('eroare label')
    
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        # print(img.shape)
        # if self.additional_transforms != None:
        #     transformed = self.additional_transforms(image = img, mask = lbl)
        #     img = transformed['image']
        #     lbl = transformed['mask']
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).float()
        # lbl = torch.from_numpy(lbl).long()
        # img = self.normalize(img)

        return img, lbl
      
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask