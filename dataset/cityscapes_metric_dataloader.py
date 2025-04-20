import lightning.pytorch as L
from .dataset_cityscapes_metric_depth import CityscapesDepthLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import albumentations as A
from torch.utils.data import DataLoader
from typing import List, Tuple

class CityscapesDataLoader(L.LightningDataModule):
    def __init__(
            self,
            base_path : str,
            batch_size: int,
            num_of_workers: int,
            augmentations: List[str],
    ) -> None:
        """
        Args:
            base_path : Path for the Cityscapes dataset.
            batch_size: Batch size for training/validation.
            augmentations: List of augmentations (currently supported: `flip`, `rotation`, `random_crop` and `increase_brightness`)
            training_internal: Flag indicating whether the training happens with internal images/open-source images.
        """
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.augmentations = augmentations
        self.ds_map = dict()
        self.input_transform = []
        self.training_transform = None
        self.validation_transform = None
        self.augmentation_dict = {'flip': A.HorizontalFlip(p = 0.5),
                                  'rotation': A.SafeRotate(),
                                  'random_scale' : A.RandomScale(scale_limit=(0.5, 2), p=1),
                                  'random_crops': A.RandomCrop(height=1024, width=1024, p=1),
                                  'increase_brightness': A.ColorJitter(brightness=(1, 3)),
                                  'blur' : A.Blur(blur_limit=(0,3))}
        self._configure_augmentations()
    
    def _configure_augmentations(self) -> None:
        """
        Method for configuring the augmentations applied at training.
        """
        if len(self.augmentations) == 0:
            return
        for augmentation in self.augmentations:
            self.input_transform.append(self.augmentation_dict[augmentation])
        self.training_transform = A.Compose(self.input_transform, is_check_shapes=False)

    def _get_dataset(self, stage: str) -> CityscapesDepthLoader:
        """
        Get dataset based on stage.

        Args:
            stage: Stage (fit/validation).
        
        Returns: 
        """
        if stage == "fit":
            return CityscapesDepthLoader(
                self.base_path,
                'train',
                True, 
                (518,518),
                additional_transforms = self.training_transform
            )
        
        return CityscapesDepthLoader(
            self.base_path,
            'val',
            True,
            (518, 518),
            additional_transforms=None
            )
    
    def setup(self, stage: str) -> None:
        """
        Setup stage for fit/validation.
        
        Args:
            stage: Stage (fit/validation).
        """
        if stage == "fit":
            self.ds_map["validation"] = self._get_dataset("validation")
        self.ds_map[stage] = self._get_dataset(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Dataloader used for training.

        Returns:
            DataLoader
        """
        return DataLoader(
            self.ds_map["fit"],
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Dataloader used for validation.

        Returns:
            DataLoader
        """
        return DataLoader(
            self.ds_map["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_of_workers
        )