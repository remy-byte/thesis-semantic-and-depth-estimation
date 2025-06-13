import hydra
from omegaconf import DictConfig
import os
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from dataset.cityscapes_segmentation_and_depth_combined_dataloader import CityscapesDataLoader
from depth_anything_v2.dpt import DepthSegmentAnythingJointV2
import torch

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def main(config: DictConfig):
    """
    Starting the training of Depth & Segmentation on Cityscapes using Hydra.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = config.trainer.visible_devices

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    datamodule = CityscapesDataLoader(
        base_path=config.data_loader.base_path,
        batch_size=config.data_loader.batch_size,
        num_of_workers=config.data_loader.num_of_workers,
        augmentations=['flip']
    )

    torch.cuda.empty_cache()

    logger = TensorBoardLogger("tb_logs", name="Cityscapes_model_joint")

    model = DepthSegmentAnythingJointV2(
        **{**model_configs[config.trainer.encoder], 'max_depth': config.trainer.max_depth},
        seg_loss_weight=1,
        depth_loss_weight=0.5
    )

    model.load_original_weights_for_training(config.trainer.original_weights)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_iou = ModelCheckpoint(
    dirpath=config.trainer.ckpt_output_path,
    filename='fullsize_1_0.5_5e-5_join_train_IOU-{epoch:02d}-{val_jaccard_epoch:.4f}-{val_abs_rel_epoch:.4f}-{val_rmse_epoch:.4f}-{val_d1_epoch:.4f}-{val_d2_epoch:.4f}-{val_d3_epoch:.4f}',
    save_top_k=1,
    monitor='val_jaccard',  
    mode='max',
)
    checkpoint_abs = ModelCheckpoint(
    dirpath=config.trainer.ckpt_output_path,
    filename='fullsize_1_0.5_5e-5_join_train_ABS_REL-{epoch:02d}-{val_jaccard_epoch:.4f}-{val_abs_rel_epoch:.4f}-{val_rmse_epoch:.4f}-{val_d1_epoch:.4f}-{val_d2_epoch:.4f}-{val_d3_epoch:.4f}',
    save_top_k=1,
    monitor='val_abs_rel',  
    mode='min', 
)
    torch.cuda.empty_cache()

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        sync_batchnorm=config.trainer.sync_batchnorm,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_model_summary=True,
        log_every_n_steps=config.data_loader.batch_size,
        logger=logger,
        callbacks=[checkpoint_iou , lr_monitor],
        strategy=config.trainer.strategy
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    main()
