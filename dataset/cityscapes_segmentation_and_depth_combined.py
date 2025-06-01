import os
import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision.transforms import Compose
from transform import PrepareForNet, Resize, NormalizeImage


class CityscapesDepthSegmentationLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(518, 518),
        depth_size=(518, 518),
        lbl_size=(518, 518),
        additional_transforms=None,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.depth_size = depth_size if isinstance(depth_size, tuple) else (depth_size, depth_size)
        self.lbl_size = lbl_size if isinstance(lbl_size, tuple) else (lbl_size, lbl_size)
        self.additional_transforms = additional_transforms
        self.n_classes = 19
        self.ignore_index = 255

        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        self.colors = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32],
        ]
        self.label_colours = dict(zip(range(19), self.colors))

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.labels_base = os.path.join(self.root, "gtFine", self.split)
        self.depth_base = os.path.join(self.root, "crestereo_depth", self.split)

        self.files = sorted(self._recursive_glob(self.images_base, suffix=".png"))
        if not self.files:
            raise Exception(f"No files found in {self.images_base} for split={split}")
        print(f"Found {len(self.files)} {split} images")

        net_h, net_w = self.img_size
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
            PrepareForNet()
        ])

    def _recursive_glob(self, rootdir=".", suffix=""):
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl_path = os.path.join(
            self.labels_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png"
        )
        lbl = cv2.imread(lbl_path, -1)
        lbl = self.encode_segmap(lbl)

        depth_name = '_'.join(os.path.basename(img_path).split('_')[:3]) + '_crestereo_depth.npy'
        depth_path = os.path.join(
            self.depth_base,
            img_path.split(os.sep)[-2],
            depth_name
        )
        depth = np.load(depth_path)

        if self.is_transform:
            img, lbl, depth = self.transform(img, lbl, depth)

        return img, depth, lbl

    def transform(self, img, lbl, depth):
        if self.additional_transforms is not None:
            transformed = self.additional_transforms(image=img, masks=[lbl, depth])
            img = transformed['image']
            lbl, depth = transformed['masks']

        img = img / 255.0
        sample = self.transform_image({'image': img, 'semseg_mask': lbl, 'depth': depth})
        img = torch.from_numpy(sample['image']).float()
        lbl = torch.from_numpy(sample['semseg_mask']).long()
        depth = torch.from_numpy(sample['depth']).float()
        return img, lbl, depth

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
