import lightning.pytorch as L
from torch.utils.data import DataLoader
from typing import List
import albumentations as A
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from .cityscapes_segmentation_and_depth_combined import CityscapesDepthSegmentationLoader
class CityscapesDataLoader(L.LightningDataModule):
    def __init__(
        self,
        base_path: str,
        batch_size: int,
        num_of_workers: int,
        augmentations: List[str],
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.augmentations = augmentations
        self.ds_map = dict()
        self.input_transform = []
        self.training_transform = None
        self.validation_transform = None

        self.augmentation_dict = {
            'flip': A.HorizontalFlip(p=0.5),
            'rotation': A.SafeRotate(),
            'random_scale': A.RandomScale(scale_limit=(0.5, 2), p=1),
            'random_crops': A.RandomCrop(height=1024, width=1024, p=1),
            'increase_brightness': A.ColorJitter(brightness=(1, 3)),
            'blur': A.Blur(blur_limit=(0, 3)),
        }

        self._configure_augmentations()

    def _configure_augmentations(self) -> None:
        if len(self.augmentations) == 0:
            return
        for augmentation in self.augmentations:
            aug = self.augmentation_dict.get(augmentation)
            if aug:
                self.input_transform.append(aug)
        self.training_transform = A.Compose(self.input_transform, is_check_shapes=False)

    def _get_dataset(self, stage: str) -> CityscapesDepthSegmentationLoader:
        split = "train" if stage == "fit" else "val"
        return CityscapesDepthSegmentationLoader(
            root=self.base_path,
            split=split,
            is_transform=True,
            img_size=(1024, 2048),
            additional_transforms=self.training_transform if stage == "fit" else None
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.ds_map["fit"] = self._get_dataset("fit")
            self.ds_map["validation"] = self._get_dataset("validation")
        elif stage == "validation":
            self.ds_map["validation"] = self._get_dataset("validation")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.ds_map["fit"],
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.ds_map["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            pin_memory=True
        )
