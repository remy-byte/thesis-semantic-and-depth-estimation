import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import lightning.pytorch as pl
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from torchmetrics.classification import MulticlassJaccardIndex
import lightning.pytorch as pl
import sys
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
import numpy as np
from torch.nn import functional as F
from torchmetrics.functional.detection import panoptic_quality
from torchmetrics.classification import JaccardIndex
import time
from torchmetrics.classification import MulticlassJaccardIndex
import math
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTSegHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        num_classes=1,
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTSegHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.num_classes = num_classes
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        
        self.scratch.head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_1, self.num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.head(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2(out)
        
        return out
    
class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


def scale_recovery(plane_est, depth, intrinsics, h_gt=1.70):
    plane_est = plane_est**3      
    b, _, h, w = depth.size()
    plane_est_down = torch.nn.functional.interpolate(plane_est.clone(),(int(h/4),int(w/4)),mode='bilinear') 
    depth_down = torch.nn.functional.interpolate(depth, (int(h/4),int(w/4)),mode='bilinear')
    int_inv = intrinsics.clone()
    int_inv[:,0:2,:] = int_inv[:,0:2,:]/4
    int_inv = int_inv.inverse()
    b, _, h, w = depth_down.size()
        
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth_down)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth_down)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth_down)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        ###pixel_coords is an array of camera pixel coordinates (x,y,1) where x,y origin is the upper left corner of the image.
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).view(b,3,-1) #.contiguous().view(b, 3, -1)  # [B, 3, H*W]

    cam_coords = int_inv.bmm(current_pixel_coords).view(b,3,h,w)
    cam_coords = cam_coords*depth_down
    cam_coords = cam_coords.reshape((b,3,-1)).permute(0,2,1) ## b, N, 3
    plane_est_down = plane_est_down.view(b,-1) ##b, N

    ## Weighted Least Squares
    W = torch.diag_embed(plane_est_down).type_as(plane_est_down)
    h = torch.ones(b,h*w,1).type_as(plane_est_down)

    left = (h.permute(0,2,1)).bmm(W).bmm(cam_coords)
    right = torch.pinverse((cam_coords.permute(0,2,1)).bmm(W).bmm(cam_coords))
    normal = (left.bmm(right)).permute(0,2,1)

    n = normal/( torch.norm(normal,dim=1).reshape(b,1,1).expand_as(normal) ) ## b,3,1 

    heights = cam_coords.bmm(n) ## b, N, 1
    height = ( (plane_est_down * heights[:,:,0]).sum(dim=1) )/(plane_est_down.sum(dim=1))

    scale_factor = ((h_gt)/height ).detach() ## scale factor is 1 if height is proper, smaller than 1 if height is too short
    # print(scale_factor)
    return scale_factor

class SSIM_Loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images."""
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Plane_Height_loss(nn.Module):
    def __init__(self, config):
        super(Plane_Height_loss, self).__init__()
        self.config = config
    
    def forward(self, plane_est, depth, intrinsics):
        scale_factor = scale_recovery(plane_est, depth, intrinsics, h_gt=self.config['camera_height'] / 30)
        target_depth = (depth.clone() * (scale_factor.reshape((-1, 1, 1, 1)).expand_as(depth))).detach()
        depth_loss = (torch.abs(target_depth - depth) / target_depth).mean()
        return scale_factor, depth_loss 

class DepthSegmentAnythingV2(pl.LightningModule):
    def __init__(self, encoder='vits', features=256, num_classes=19, out_channels=[48, 96, 192, 384], 
                 use_bn=False, use_clstoken=False, max_depth=80.0, optimizer_type="ADAMW", base_lr=0.001, 
                 weight_decay=1e-4, momentum=None, nesterov=False, scheduler_name='ReduceLRonPlateau', 
                 max_iters=17336):
        super(DepthSegmentAnythingV2, self).__init__()
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.scheduler_name = scheduler_name
        self.max_iters = max_iters
        
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 
                                                0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
                                                1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 
                                                1.0507])
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.segmentation_head = DPTSegHead(self.pretrained.embed_dim, features, num_classes=self.num_classes, use_bn=use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
        # Instantiate loss functions
        self.ssim = SSIM_Loss()
        self.plane_loss = Plane_Height_loss({'camera_height': 1.6, 'l_scale_recovery': True})
        self.l_smooth_weight = 0.1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)  # or any other optimizer
        return optimizer

    def configure_optimizers(self):
        params = self.parameters()
        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(params, lr=self.base_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        elif self.optimizer_type == "ADAM":
            optimizer = optim.Adam(params, lr=self.base_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "ADAMW":
            optimizer = optim.AdamW(params, lr=self.base_lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"No optimizer type {self.optimizer_type}")

        if self.scheduler_name == "ReduceLRonPlateau":
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)
            return [optimizer], [{"scheduler": scheduler, "name": "ReduceOnPlateauScheduler", "interval": "epoch", "monitor": "val_loss", "frequency": 1}]
        elif self.scheduler_name == "PolyLR":
            scheduler = PolynomialLR(optimizer=optimizer, total_iters=self.max_iters, power=1/2)
            return [optimizer], [{"scheduler": scheduler, "name": "PolyLRScheduler", "interval": "step", "frequency": 1}]
        else:
            return [optimizer]

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        segmentation, depth = self.forward(image)
        plane_est = segmentation[:, :, :, 1].detach()  # Assuming class index 1 represents the ground plane

        # Compute losses
        loss_smooth = self.get_smooth_loss(depth, image)
        scale_factor, loss_scale_depth = self.plane_loss(plane_est, depth, torch.eye(3).to(image.device))
        
        total_loss = loss_smooth + loss_scale_depth
        
        self.log('train_loss', total_loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        segmentation, depth = self.forward(image)
        plane_est = segmentation[:, 0, :, :].detach()  # Assuming class index 1 represents the ground plane

        # Compute losses
        loss_smooth = self.get_smooth_loss(depth, image)
        scale_factor, loss_scale_depth = self.plane_loss(plane_est, depth, torch.eye(3).to(image.device))

        total_loss = loss_smooth + loss_scale_depth
        
        self.log('val_loss', total_loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return total_loss

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)  # No need to multiply by max_depth
        segmentation = self.segmentation_head(features, patch_h, patch_w)
        return segmentation, depth.squeeze(1)

    def get_smooth_loss(self, disp, img):
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()
    

