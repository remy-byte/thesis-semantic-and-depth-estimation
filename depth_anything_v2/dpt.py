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


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        max_depth=20.0
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.max_depth = max_depth
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    
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

# class DepthAnythingV2(nn.Module):
#     def __init__(
#         self, 
#         encoder='vitl', 
#         features=256, 
#         out_channels=[256, 512, 1024, 1024], 
#         use_bn=False, 
#         use_clstoken=False
#     ):
#         super(DepthAnythingV2, self).__init__()
        
#         self.intermediate_layer_idx = {
#             'vits': [2, 5, 8, 11],
#             'vitb': [2, 5, 8, 11], 
#             'vitl': [4, 11, 17, 23], 
#             'vitg': [9, 19, 29, 39]
#         }
        
#         self.encoder = encoder
#         self.pretrained = DINOv2(model_name=encoder)
        
#         self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
#     def forward(self, x):
#         patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
#         features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
#         # for feat in features:
#         #     for f in feat:
#         #         print(f.shape)
        
    
#         depth = self.depth_head(features, patch_h, patch_w)
#         depth = F.relu(depth)
        
#         return depth.squeeze(1)
    
#     @torch.no_grad()
#     def infer_image(self, raw_image, input_size=518):
#         image, (h, w) = self.image2tensor(raw_image, input_size)
        
#         depth = self.forward(image)
        
#         depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
#         return depth.cpu().numpy()
    
#     def image2tensor(self, raw_image, input_size=518):        
#         transform = Compose([
#             Resize(
#                 width=input_size,
#                 height=input_size,
#                 resize_target=False,
#                 keep_aspect_ratio=True,
#                 ensure_multiple_of=14,
#                 resize_method='lower_bound',
#                 image_interpolation_method=cv2.INTER_CUBIC,
#             ),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             PrepareForNet(),
#         ])
        
#         h, w = raw_image.shape[:2]
        
#         image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
#         image = transform({'image': image})['image']
#         image = torch.from_numpy(image).unsqueeze(0)
        
#         DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#         image = image.to(DEVICE)
        
#         return image, (h, w)


class DepthSegmentAnythingV2(pl.LightningModule):
    def __init__(
        self, 
        encoder='vits', 
        features=256, 
        num_classes = 19,
        out_channels=[48, 96, 192, 384], 
        use_bn=False, 
        use_clstoken=False,
        max_depth = 80.0,
        optimizer_type="ADAMW",
        base_lr=0.00006, 
        weight_decay=1e-4,
        momentum=None,
        nesterov=False, 
        scheduler_name='PolyLR',
        max_iters=18575
        
    ):
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.scheduler_name = scheduler_name
        self.max_iters = max_iters
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507])
        super(DepthSegmentAnythingV2, self).__init__()
        
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


    def configure_optimizers(self):
        # Instantiate and configure the optimizers
        params = self.parameters()
        
        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(
                params,
                lr=self.base_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif self.optimizer_type == "ADAM":
            optimizer = optim.Adam(
                params,
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAMW":
            optimizer = optim.AdamW(
                params,
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"No optimizer type {self.optimizer_type}")

        if self.scheduler_name == "ReduceLRonPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            return [optimizer], [{"scheduler": scheduler, "name": "ReduceOnPlateauScheduler", "interval": "epoch", "monitor": "val_loss", "frequency": 1}]
        
        elif self.scheduler_name == "PolyLR":
            scheduler = PolynomialLR(
                optimizer=optimizer,
                total_iters=self.max_iters,
                power=1/2
            )
            return [optimizer], [{"scheduler": scheduler, "name": "PolyLRScheduler", "interval": "step", "frequency": 1}]
        
        else:
            return [optimizer]


 

    def load_original_weights_for_training(self, file: str, freeze_encoder=True):
        '''
        Loading the weights from the original pth that contains only the weights for the original encoder and decoder for depth.
        Additional step done, by loading the depth weights from the decoder into the segmentation decoder as a form of  "pretraining" 
        '''
        # loading the original encoder and decoder of the arhitecture
        self.state = torch.load(file, map_location= 'cpu')
        self.load_state_dict(self.state, strict=False)

        # for the segmentation we will also load the same decoder used for depth for the sake of having it "pretrained"

        new_state = {}
        for param in self.state.keys():
            new_key = param.replace('depth_head.', '')
            new_state[new_key] = self.state[param]
        
        # taking out the last two conv layers
        new_state.popitem()
        new_state.popitem()
        new_state.popitem()

        # self.segmentation_head.load_state_dict(new_state, strict=False)
        for param in self.depth_head.parameters():
            param.requires_grad = False

        if freeze_encoder == True:
            for param in self.pretrained.parameters(): 
                param.requires_grad = False

        

    def get_segmentation_metrics(self, gt_label, pred_label):
        js = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(self.device)
        res = js(pred_label, gt_label)
        return res
    
    def get_segmentation_loss(self, predictions, targets, ignore_value=255, loss_weight=1.0):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        loss = F.cross_entropy(
            predictions.to(self.device), targets.to(self.device), reduction="mean", ignore_index=ignore_value, weight=self.class_weights.to(self.device)).to(self.device)
        return loss * loss_weight
    
         
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        # image, (h, w) = self.image2tensor(image, 518)
        label = label
        segmentation, _ = self.forward(image)
        loss = self.get_segmentation_loss(segmentation,label)
        pred = torch.argmax(segmentation, dim=1).float()
        metric = self.get_segmentation_metrics(label, pred)
        

        self.log('train_loss',
                 loss,
                 prog_bar = True,
                 on_epoch = True,
                 on_step = True,
                 sync_dist = True)
       
        self.log('train_jaccard',
                 metric,
                 prog_bar = True,
                 on_epoch = True,
                 on_step = True,
                 sync_dist = True)
 
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        # image, (h, w) = self.image2tensor(image, 518)
        label = label
        segmentation, _ = self.forward(image)
        loss = self.get_segmentation_loss(segmentation,label)
        pred = torch.argmax(segmentation, dim=1).float()
        metric = self.get_segmentation_metrics(label, pred)
        
        self.log('val_loss',
                 loss,
                 prog_bar = True,
                 on_epoch = True,
                 on_step = True,
                 sync_dist = True)
       
       
        self.log('val_jaccard',
                 metric,
                 prog_bar = True,
                 on_epoch = True,
                 on_step = True,
                 sync_dist = True)
 
        return loss
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        # for feat in features:
        #     for f in feat:
        #         print(f.shape)
        
    
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        depth = F.relu(depth)
        segmentation = self.segmentation_head(features, patch_h, patch_w) 
        
        return segmentation, depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        segmentation, depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return segmentation, depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    


class DepthMetricAnything(pl.LightningModule):
    def __init__(
        self, 
        encoder='vits', 
        features=256, 
        num_classes = 19,
        out_channels=[48, 96, 192, 384], 
        use_bn=False, 
        use_clstoken=False,
        max_depth = 80.0,
        optimizer_type="ADAMW",
        base_lr=0.00005, 
        weight_decay=1e-4,
        momentum=None,
        nesterov=False, 
        scheduler_name='PolyLR',
        max_iters=12383
        
    ):
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.scheduler_name = scheduler_name
        self.max_iters = max_iters
        super(DepthMetricAnything, self).__init__()
        
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



    def configure_optimizers(self):
        # Instantiate and configure the optimizers
        params = self.parameters()
        
        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(
                params,
                lr=self.base_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif self.optimizer_type == "ADAM":
            optimizer = optim.Adam(
                params,
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAMW":
            optimizer = optim.AdamW(
                params,
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"No optimizer type {self.optimizer_type}")

        if self.scheduler_name == "ReduceLRonPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            return [optimizer], [{"scheduler": scheduler, "name": "ReduceOnPlateauScheduler", "interval": "epoch", "monitor": "val_loss", "frequency": 1}]
        
        elif self.scheduler_name == "PolyLR":
            scheduler = PolynomialLR(
                optimizer=optimizer,
                total_iters=self.max_iters,
                power=1/2
            )
            return [optimizer], [{"scheduler": scheduler, "name": "PolyLRScheduler", "interval": "step", "frequency": 1}]
        
        else:
            return [optimizer]


 

    def load_original_weights_for_training(self, file: str):
        '''
        Loading the weights from the original pth that contains only the weights for the original encoder and decoder for depth.
        Additional step done, by loading the depth weights from the decoder into the segmentation decoder as a form of  "pretraining" 
        '''
        self.state = torch.load(file)
        self.load_state_dict(self.state, strict=True)

        

    def get_depth_metrics(self, pred, target):
        assert pred.shape == target.shape

        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)
        d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
        d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)

        abs_rel = torch.mean(torch.abs(diff) / target)
        sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
                'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}
        
    def get_depth_loss(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          0.5 * torch.pow(diff_log.mean(), 2))

        return loss
    
         
    def training_step(self, train_batch, batch_idx):
        image, label, valid_mask = train_batch
        # image, (h, w) = self.image2tensor(image, 518)
        label = label
        depth = self.forward(image)
        loss = self.get_depth_loss(depth, label, (valid_mask == 1) & (label >= 0.001) & (label <= 80.0))
        metric = self.get_depth_metrics(depth[valid_mask], label[valid_mask])
        

        self.log('train_loss', 
                loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True, 
                sync_dist=True) 
        for metric_name, metric_value in metric.items(): 
            self.log(f'train_{metric_name}',
                metric_value, 
                prog_bar=True, 
                on_epoch=True, 
                on_step=True, 
                sync_dist=True)
            
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        image, label, valid_mask = val_batch
        # image, (h, w) = self.image2tensor(image, 518)
        label = label
        depth = self.forward(image)
        loss = self.get_depth_loss(depth, label, (valid_mask == 1) & (label >= 0.001) & (label <= 80.0))
        metric = self.get_depth_metrics(depth[valid_mask], label[valid_mask])
        
        self.log('val_loss', 
                loss,
                prog_bar=True,
                on_epoch=True,
                on_step=True, 
                sync_dist=True) 
        for metric_name, metric_value in metric.items(): 
            self.log(f'val_{metric_name}',
                metric_value, 
                prog_bar=True, 
                on_epoch=True, 
                on_step=True, 
                sync_dist=True)
            
        return loss
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
    
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

