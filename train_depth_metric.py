import hydra
from omegaconf import DictConfig
import sys
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import os
from dataset.cityscapes_metric_dataloader import CityscapesDataLoader  
import optparse as opts
from depth_anything_v2.dpt import DepthMetricAnything
import torch

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def main(config: DictConfig):
    """
        Starting the training of Panoptic DeepLab on SemSeg Dataset.
    """
 
    # make devices visible
    os.environ['CUDA_VISIBLE_DEVICES']=config.trainer.visible_devices
 
    # cfg = _setup(config.trainer.initial_weights, config.model.backbone)

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
    
    torch.autograd.set_detect_anomaly(True)
    datamodule = CityscapesDataLoader(
        base_path = config.data_loader.base_path,
        batch_size= config.data_loader.batch_size, 
        num_of_workers= config.data_loader.num_of_workers,
        augmentations= ['flip']
        )
    # torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    logger = TensorBoardLogger("tb_logs", name="metric_model")
    model = DepthMetricAnything(**{**model_configs[config.trainer.encoder], 'max_depth': config.trainer.max_depth})
    model.load_original_weights_for_training(config.trainer.original_weights)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    max_iou_callback = ModelCheckpoint(
        dirpath=config.trainer.ckpt_output_path,
        filename='100epochs_metric_depth_cityscapes_vitb_{epoch}-{train_loss_epoch:.2f}-{val_loss_epoch:.2f}-{val_d1_epoch:.2f}-{val_d2_epoch:.2f}-{val_d3_epoch:.2f}-{val_abs_rel_epoch:.2f}-{val_sq_rel_epoch:.2f}-{val_rmse_epoch:.2f}-{val_log10_epoch:.2f}-{val_silog_epoch:.2f}',
        save_top_k=1,
        mode="min",
        monitor="val_abs_rel",
        every_n_epochs=1, 
    )

    trainer = pl.Trainer(
        max_epochs = config.trainer.max_epochs,
        accelerator = config.trainer.accelerator,
        devices = config.trainer.devices,
        precision = config.trainer.precision,
        sync_batchnorm = config.trainer.sync_batchnorm,
        enable_checkpointing = config.trainer.enable_checkpointing,
        enable_model_summary = True,
        log_every_n_steps=config.data_loader.batch_size,
        logger = logger,
        callbacks=[max_iou_callback, lr_monitor],
        strategy=config.trainer.strategy,
    
        
    )

    trainer.fit(model = model, datamodule = datamodule)    


if __name__ == '__main__':
    main()