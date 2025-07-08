# Copyright (c) Zhijia Technology. All rights reserved.

# Author: Peidong Li

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.utils import LearnedPositionalEncoding
from .view_transformer import DepthNet, LSSViewTransformerBEVDepth, LSSViewTransformerBEVStereo
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import matplotlib.pyplot as plt
import open3d as o3d
import pdb
import random
import cv2

from .MVSGaussian.lib.gaussian_renderer import render

from .MVSGaussian.lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss
from .MVSGaussian.lib.train.losses.ssim_loss import SSIM
from .MVSGaussian.lib.utils import data_utils, graphics_utils

from ...ops.cross_attention_2d import DeformableAttention2D
from ...ops.dynamic_conv import Dynamic_conv2d, Dynamic_conv3d


class MS_CAM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"
    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)

class ChannelAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes,planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ProbNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(ProbNet, self).__init__()
        self.loss_weight=loss_weight
        self.loss_weight_opacity = 6.0
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.prob_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),      
            )
        self.mask_net = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0],bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.dice_loss_opacity = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight_opacity)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        height_feat = self.base_conv(input)
        height_feat = self.prob_conv(height_feat)            
        bev_prob = self.mask_net(height_feat)            
        return bev_prob

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h,1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)
    
    def get_bev_opacity_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_opacity_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h,1)).mean()
        else:
            mask_ce_opacity_loss = self.ce_loss(a, b)
        mask_dice_opacity_loss = self.dice_loss_opacity(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_opacity_loss=self.loss_weight_opacity*mask_ce_opacity_loss, mask_dice_opacity_loss=mask_dice_opacity_loss)

class DualFeatFusion(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(DualFeatFusion, self).__init__()
        self.ca = MS_CAM(input_channel, output_channel)
    
    def forward(self, x1, x2):
        channel_factor = self.ca(torch.cat((x1,x2),1))
        out = channel_factor*x1 + (1-channel_factor)*x2

        return out

class BEVGeomAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVGeomAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1+bev_prob)

class ObatinOpacityMask(nn.Module):
    def __init__(self, kernel_size=7):
        super(ObatinOpacityMask, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, opacity_bev):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        out2 = self.sigmoid(x + opacity_bev)
        return out2

def bev_centerness_weight(nx, ny):
    xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, nx))
    grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
    grid = grid - nx//2
    grid = grid / (nx//2)
    centerness = (grid[..., 0]**2 + grid[..., 1]**2) / 2 
    centerness = centerness.sqrt() + 1
    return centerness

class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, loss_weight=1.):
        super(DiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.loss_weight=loss_weight

    def forward(self, inputs, targets, smooth=1e-5):
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)     
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return self.loss_weight*(1 - dice)

class ScaleFactorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ScaleFactorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softplus(x)
        return x
    
class RotationFactorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RotationFactorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

class OpacityFactorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OpacityFactorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class ColorFactorMLPGaussian(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ColorFactorMLPGaussian, self).__init__()
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class ColorFactorMLPNerf(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1):
        super(ColorFactorMLPNerf, self).__init__()
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
class DepthFactorMLPNerf(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1):
        super(DepthFactorMLPNerf, self).__init__()
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class ImgFeatResize1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1):
        super(ImgFeatResize1, self).__init__()
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
class ImgFeatResize2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1):
        super(ImgFeatResize2, self).__init__()
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class UNetLight(nn.Module):
    def __init__(self):
        super(UNetLight, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)    
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 80, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(80, 80, kernel_size=2, stride=2) 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x

class ScaledSwish(nn.Module):
    def __init__(self):
        super(ScaledSwish, self).__init__()

    def forward(self, x):
        swish = x / (1 + torch.exp(-x))
        return 2 * swish - 1 

class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh(2.0 / 3.0 * x)

class HeightAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(HeightAttention, self).__init__()

        input_channel = input_channel // 4
        output_channel = output_channel // 4

        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        
        self.max_pool2 = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        
        self.max_pool3 = nn.AdaptiveMaxPool2d(1)
        self.conv3 = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        
        self.max_pool4 = nn.AdaptiveMaxPool2d(1)
        self.conv4 = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        height_slice = int(x.shape[1] / 4)
        max_out1 = self.conv1(self.max_pool1(x[:, 0:height_slice, :, :]))
        max_out2 = self.conv2(self.max_pool2(x[:, height_slice:height_slice*2, :, :]))
        max_out3 = self.conv3(self.max_pool3(x[:, height_slice*2:height_slice*3, :, :]))
        max_out4 = self.conv4(self.max_pool4(x[:, height_slice*3:height_slice*4, :, :]))

        output = torch.cat([max_out1, max_out2, max_out3, max_out4], dim=1)
        output1 = self.tanh(output)
        
        return output1

class OpacityVoxelToBEVConverter(nn.Module):
    def __init__(self, input_channel=13):
        super(OpacityVoxelToBEVConverter, self).__init__()
        self.encoder1 = self.conv_block(in_channels=input_channel, out_channels=4)
        self.ca1 = HeightAttention(4, 4, 1)
        self.encoder2 = self.conv_block(in_channels=4, out_channels=8)
        self.ca2 = HeightAttention(8, 8, 1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self.conv_block(in_channels=8, out_channels=16)
        self.ca_bottleneck = HeightAttention(16, 16, 1)
        
        self.upconv2 = self.upconv(in_channels=16, out_channels=8)
        self.decoder2 = self.conv_block(in_channels=16, out_channels=8)
        self.ca_dec2 = HeightAttention(8, 8, 1)
        
        self.upconv1 = self.upconv(in_channels=8, out_channels=4)
        self.decoder1 = self.conv_block(in_channels=8, out_channels=4)
        self.ca_dec1 = HeightAttention(4, 4, 1)
        
        self.output_conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), 
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, position):
        enc1 = self.encoder1(x) + position
        enc1 = self.ca1(enc1) * enc1
        enc2 = self.encoder2(self.pool(enc1))
        enc2 = self.ca2(enc2) * enc2
        
        bottleneck = self.bottleneck(self.pool(enc2))
        bottleneck = self.ca_bottleneck(bottleneck) * bottleneck
        
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.ca_dec2(dec2) * dec2
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.ca_dec1(dec1) * dec1
        
        out = self.output_conv(dec1)
        
        return out

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, in_planes=1, out_planes=13):
        super(VoxelFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class ResizeNetwork(nn.Module):
    def __init__(self, output_channel):
        super(ResizeNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=32, out_channels=output_channel, kernel_size=2, stride=2)
        
        self.upsample3 = nn.ConvTranspose2d(in_channels=output_channel, out_channels=output_channel, kernel_size=4, stride=4)
        
    def forward(self, x):
        x = self.conv1(x) 
        x = self.upsample1(x) 
        
        x = self.conv2(x)
        x = self.upsample2(x)
        
        x = self.upsample3(x)
        
        return x

class LinearWeightedImage(nn.Module):
    def __init__(self):
        super(LinearWeightedImage, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, a1, a2):
        a = self.w * a1 + (1 - self.w) * a2
        return a
    
class LinearWeightedDepth(nn.Module):
    def __init__(self):
        super(LinearWeightedDepth, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, a1, a2):
        a = self.w * a1 + (1 - self.w) * a2
        return a

iteration = 0

@NECKS.register_module()
class OcRFViewTransformerFull(LSSViewTransformerBEVStereo):
    def __init__(self, pc_range, bev_h=128, bev_w=128, num_height=13, collapse_z=True, loss_semantic_weight=25,
                 depth_threshold=1, semantic_threshold=0.25, depthnet_cfg=dict(), **kwargs):
        super(OcRFViewTransformerFull, self).__init__(**kwargs)
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_threshold = depth_threshold / self.D
        self.semantic_threshold = semantic_threshold
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_height = num_height
        self.collapse_z = collapse_z

        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                            self.out_channels, self.D+2, **depthnet_cfg)
        self.fuser = DualFeatFusion(2*self.out_channels,self.out_channels)
        self.geom_att = BEVGeomAttention()
        self.ObatinOpacityMask = ObatinOpacityMask()
        self.prob = ProbNet(in_channels=self.out_channels, with_centerness=True, bev_size=(self.bev_h,self.bev_w))
        self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.bev_h, self.bev_w)
        self.positional_encoding1 = LearnedPositionalEncoding(8 // 4, self.bev_h, self.bev_w)

        # nerf
        input_dim = 80
        hidden_dim1 = 4
        output_dim_c = 3
        output_dim_d = 1
        self.image_feat_resize = ResizeNetwork(input_dim)
        self.sigma = nn.Sequential(nn.Linear(input_dim, hidden_dim1), nn.Linear(hidden_dim1, 1), nn.Softplus())
        self.C_MLP_nerf = ColorFactorMLPNerf(input_dim, hidden_dim1, output_dim_c)
        self.D_MLP_nerf = DepthFactorMLPNerf(input_dim, hidden_dim1, output_dim_d)
        self.img_feat_resize1 = ImgFeatResize1(input_dim, hidden_dim1, output_dim_c)
        self.img_feat_resize2 = ImgFeatResize2(input_dim, hidden_dim1, output_dim_d)

        input_dim = 80 
        hidden_dim = 4
        output_dim_s = 3 
        output_dim_r = 4
        output_dim_a = 1
        output_dim_c = 3

        # gaussian
        self.S_MLP = ScaleFactorMLP(input_dim, hidden_dim, output_dim_s)
        self.R_MLP = RotationFactorMLP(input_dim, hidden_dim, output_dim_r)
        self.A_MLP = OpacityFactorMLP(input_dim, hidden_dim, output_dim_a)
        self.C_MLP = ColorFactorMLPGaussian(input_dim, hidden_dim, output_dim_c)


        self.OpacityVoxelToBEV = OpacityVoxelToBEVConverter(input_channel=13)

        self.color_crit = nn.MSELoss(reduction='mean')

        self.zfar = 999.9
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

        self.ObtainVoxelFeature = VoxelFeatureExtractor()

        self.LinearWeightedImage = LinearWeightedImage()
        self.LinearWeightedDepth = LinearWeightedDepth()

        self.defor_cross_attention = DeformableAttention2D(
                                        dim = 13,
                                        dim_head = 8,
                                        heads = 1,
                                        dropout = 0.1,
                                        downsample_factor = 4,
                                        offset_scale = 4,
                                        offset_groups = None,
                                        offset_kernel_size = 6
                                    )
        

    def get_reference_points_3d(self, H, W, Z=8, num_points_in_pillar=13, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in HT.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in HT, has \
                shape (bs, D, HW, 3).
        """
        zs_l = torch.linspace(3, Z-1, 5, dtype=dtype,device=device)
        zs_g = torch.linspace(0.5, Z - 0.5, num_points_in_pillar-5, dtype=dtype,device=device)
        zs = torch.cat((zs_l,zs_g)).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    @force_fp32()
    def get_projection(self, rots, trans, intrins, post_rots, post_trans, bda):
        B, N, _, _ = rots.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        inv_sensor2ego = torch.inverse(rots)
        lidar2img_R = intrins.matmul(inv_sensor2ego).matmul(torch.inverse(bda)) # [B, 6, 3, 3]
        lidar2img_t = -intrins.matmul(inv_sensor2ego).matmul(trans.unsqueeze(-1)) # [B, 6, 3, 1]

        lidar2img = torch.cat((lidar2img_R, lidar2img_t), -1) 
        img_aug = torch.cat((post_rots, post_trans.unsqueeze(-1)), -1)
        return lidar2img, img_aug, lidar2img_R, lidar2img_t

    @force_fp32()
    def get_sampling_point(self, reference_points, pc_range, depth_range, lidar2img, img_aug, image_shapes):
        # B, bev_z, bev_h* bev_w, 3
        reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

        # B, D, HW, 3
        B, Z, num_query = reference_points.size()[:3]
        reference_points = reference_points.view(B, -1 , 3)
        num_cam = lidar2img.size(1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        pseudo_point_cloud_lidar = reference_points.clone()

        reference_points = reference_points.view(B, 1, Z*num_query, 4).repeat(1, num_cam, 1, 1)

        lidar2img = lidar2img.view(B, num_cam, 1, 3, 4)
        img_aug = img_aug.view(B, num_cam, 1, 3, 4)

        reference_points = lidar2img.matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        referenece_depth = reference_points[..., 2:3].clone()
        bev_mask = (reference_points[..., 2:3] > eps)

        reference_points_cam = torch.cat((reference_points[..., 0:2] / torch.maximum(
            reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])*eps), reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])), -1)

        reference_points_cam = torch.matmul(img_aug, 
                                        reference_points_cam.unsqueeze(-1)).squeeze(-1)  
    
        reference_points_cam = reference_points_cam[..., 0:2]
        reference_points_cam[..., 0] /= image_shapes[1]
        reference_points_cam[..., 1] /= image_shapes[0]

        reference_points_cam= reference_points_cam.view(B,num_cam,Z,num_query,2)
        referenece_depth= referenece_depth.view(B,num_cam,Z,num_query,1)
        bev_mask= bev_mask.view(B,num_cam,Z,num_query,1)

        bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > 0.0) 
                    & (reference_points_cam[..., 0:1] < 1.0) 
                    & (reference_points_cam[..., 1:2] > 0.0) 
                    & (reference_points_cam[..., 1:2] < 1.0))
        # D, B, N, num_query, 1
        if depth_range is not None:
            referenece_depth = (referenece_depth-depth_range[0])/(depth_range[1]-depth_range[0])
            bev_mask = (bev_mask & (referenece_depth > 0.0)
                        & (referenece_depth < 1.0))
        # [B, N, Z*Nq, 1] bev_mask
        # [B, N, Z*Nq, 2] reference_points_cam
        # [B, N, Z*Nq, 1] referenece_depth
        bev_mask = torch.nan_to_num(bev_mask)
        return torch.cat((reference_points_cam, referenece_depth),-1), bev_mask, [pseudo_point_cloud_lidar, reference_points_cam, lidar2img]
        # return torch.cat((reference_points_cam, referenece_depth),-1), bev_mask

    def init_acceleration_ht(self, coor, mask):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)

        self.ranks_bev_ht = ranks_bev.int().contiguous()
        self.ranks_feat_ht = ranks_feat.int().contiguous()
        self.ranks_depth_ht = ranks_depth.int().contiguous()
        self.interval_starts_ht = interval_starts.int().contiguous()
        self.interval_lengths_ht = interval_lengths.int().contiguous()

    def fast_sampling(self, coor, mask, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                1, int(self.bev_h), int(self.bev_w)
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)

        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)

        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def fast_sample_prepare(self, coor, mask):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the image space in
                shape (B, N, ZNq 3).
            mask (torch.tensor): mask of points in the imaage space in
                shape (B, N, ZNq, 1).
        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, Z, Nq, _ = coor.shape
        num_points = B * N * Z * Nq
        # record the index of selected points for acceleration purpose
        ranks_bev = torch.range(
            0, num_points // (N*Z) - 1, dtype=torch.int, device=coor.device)
        ranks_bev = ranks_bev.reshape(B, 1, 1, Nq)
        ranks_bev = ranks_bev.expand(B, N, Z, Nq).flatten()
        # convert coordinate into the image feat space
        coor[..., 0] *= self.W
        coor[..., 1] *= self.H
        coor[..., 2] *= self.D
        # [B, N, Z, Nq, 3]
        coor = coor.round().long().view(num_points, 3)
        coor[..., 0].clamp_(min=0, max=self.W-1)
        coor[..., 1].clamp_(min=0, max=self.H-1)
        coor[..., 2].clamp_(min=0, max=self.D-1)
        batch_idx = torch.range(0, B*N-1).reshape(B*N, 1). \
            expand(B*N, num_points // (B*N)).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = mask.reshape(-1)
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_bev = \
            coor[kept], ranks_bev[kept]

        ranks_depth = coor[:, 3] * (self.D * self.W * self.H)
        ranks_depth += coor[:, 2] * (self.W * self.H)
        ranks_depth += coor[:, 1] * self.W + coor[:, 0]
        depth_size = B * N * self.D * self.W * self.H
        ranks_depth.clamp_(min=0, max=depth_size-1)

        ranks_feat = coor[:, 3] * (self.W * self.H)
        ranks_feat += coor[:, 1] * self.W + coor[:, 0]
        feat_size = B * N * self.W * self.H

        ranks_feat.clamp_(min=0, max=feat_size-1)

        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.lidar2img, self.img_aug, _, _ = self.get_projection(*input[1:7])
            self.init_acceleration_v2(coor)

            self.W = self.input_size[1] / self.downsample
            self.H = self.input_size[0] / self.downsample
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, num_points_in_pillar=self.num_height, bs=1)
            coor, mask, _ = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], self.lidar2img, self.img_aug, self.input_size)
            self.init_acceleration_ht(coor, mask)

            self.initial_flag = False

    def get_lss_bev_feat(self, input, depth, tran_feat, kept=None):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat

    def get_ht_bev_feat(self, input, depth, tran_feat, bev_mask=None):
        B, N, C, H, W = input[0].shape
        self.H = H
        self.W = W

        # Prob-Sampling
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth_ht,
                                   self.ranks_feat_ht, self.ranks_bev_ht,
                                   bev_feat_shape, self.interval_starts_ht,
                                   self.interval_lengths_ht)
            if bev_mask is not None:
                bev_feat = bev_feat * bev_mask
            bev_feat = bev_feat.squeeze(2)
        else:
            lidar2img, img_aug, lidar2img_R, lidar2img_t = self.get_projection(*input[1:7])
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, bs=B, num_points_in_pillar=self.num_height)
            coor, mask, [pseudo_point_cloud_lidar, pseudo_point_cloud_cam, lidar2img] = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], lidar2img, img_aug, self.input_size) 

            if bev_mask is not None:
                mask = bev_mask * mask.view(B,N,self.num_height,self.bev_h,self.bev_w)
            bev_feat = self.fast_sampling(
                coor, mask, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, voxel, [pseudo_point_cloud_lidar, pseudo_point_cloud_cam, mask, lidar2img, lidar2img_R, lidar2img_t]

    def lidar_points_to_image_values(self, pillars, imgs, mask):
        B, num_cams, num_points_per_pillar, num_pillars, _ = pillars.size()
        _, _, C, H, W = imgs.size()

        pillars_normalized = pillars.clone()
        pillars_normalized[..., 0] = (pillars[..., 0] / (W - 1)) * 2 - 1 
        pillars_normalized[..., 1] = (pillars[..., 1] / (H - 1)) * 2 - 1 

        pillars_normalized = pillars_normalized.float()
        pillars_normalized = pillars_normalized.view(B * num_cams, num_points_per_pillar * num_pillars, 2)
        imgs = imgs.view(B * num_cams, C, H, W)

        pillars_normalized = pillars_normalized.unsqueeze(1) 
        img_values = F.grid_sample(imgs.float(), pillars_normalized, align_corners=True)

        img_values = img_values.view(B, num_cams, C, num_points_per_pillar, num_pillars).permute(0, 1, 3, 4, 2)
        img_values = img_values * mask.float()

        return img_values


    def color_voxels(self, voxels, img_values, mask, downsample_factor_Z=None, downsample_factor_XY=None):
        B, P, N, _ = voxels.shape
        num_cam = img_values.shape[1]

        mask_expanded = mask.squeeze(-1)
        valid_mask = mask_expanded.any(dim=1)

        img_values_masked = torch.where(mask_expanded[..., None], img_values, torch.zeros_like(img_values))
        valid_counts = mask_expanded.sum(dim=1, keepdim=True).float()  # [B, 1, P, N]
        valid_counts = valid_counts.unsqueeze(-1)  # [B, 1, P, N, 1]
        valid_counts = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)

        avg_color = img_values_masked.sum(dim=1, keepdim=True) / valid_counts  # [B, 1, P, N, 3]
        avg_color = avg_color.squeeze(1)  # [B, P, N, 3]

        colored_voxels = torch.zeros((B, P, N, 6), device=voxels.device)  # [B, P, N, 6]
        colored_voxels[..., :3] = voxels
        colored_voxels[..., 3:] = avg_color

        if downsample_factor_XY != None:
            colored_voxels = colored_voxels[:, :, ::downsample_factor_XY, :]  # [B, P, N//downsample_factor, 6]
            valid_mask = valid_mask[:, :, ::downsample_factor_XY]  # [B, P, N//downsample_factor]
        if downsample_factor_Z != None:
            colored_voxels = colored_voxels[:, ::downsample_factor_Z, :, :]  # [B, P, N//downsample_factor, 6]
            valid_mask = valid_mask[:, ::downsample_factor_Z, :]  # [B, P, N//downsample_factor]

        return colored_voxels, avg_color, valid_mask

    def compute_parameters(self, B, camera_intrinsics, camera_extrinsics, image_width, image_height):
        """
        Computes the FOV, world view transform, full projection transform, and camera center.

        Args:
            B (int): Batch size
            camera_intrinsics (torch.Tensor): Camera intrinsic parameters of shape [B, 6, 3, 3]
            camera_extrinsics (torch.Tensor): Camera extrinsic parameters of shape [B, 6, 3, 4]
            image_width (int): Width of the image
            image_height (int): Height of the image

        Returns:
            fov_x, fov_y, world_view_transform, full_proj_transform, camera_center
        """
        focal_lengths_x = camera_intrinsics[..., 0, 0]
        fov_x = 2 * torch.atan(image_width / (2 * focal_lengths_x)) 
        
        focal_lengths_y = camera_intrinsics[..., 1, 1]
        fov_y = 2 * torch.atan(image_height / (2 * focal_lengths_y))

        world_view_transform = camera_extrinsics[..., :3, :4]

        full_proj_transform = torch.zeros((B, 6, 4, 4), device=camera_intrinsics.device)
        full_proj_transform[..., :3, :3] = camera_intrinsics
        full_proj_transform[..., :3, 3] = camera_extrinsics[..., :3, 3]
        full_proj_transform[..., 3, 3] = 1

        camera_center = camera_intrinsics[..., :2, 2]

        return fov_x, fov_y, world_view_transform, full_proj_transform, camera_center

    def retain_valid_pixels(self, image_matrix, pseudo_point_cloud, mask):
        batch_size, num_images, _, img_height, img_width = image_matrix.shape
    
        result_matrix = torch.ones_like(image_matrix) * 255
        
        mask = mask.expand(-1, -1, -1, -1, -1, 2)

        valid_coords = torch.where(mask, pseudo_point_cloud, torch.tensor(-1, device=pseudo_point_cloud.device))

        for b in range(batch_size):
            for n in range(num_images):
                for z in range(pseudo_point_cloud.shape[2]):
                    coords = valid_coords[b, n, z]
                    mask_slice = (coords[..., 0] != -1)
                    clamped_coords = torch.clamp(coords[mask_slice].long(), min=0, max=max(img_width, img_height) - 1)

                    y_coords, x_coords = clamped_coords[:, 1], clamped_coords[:, 0]
                    
                    result_matrix[b, n, :, y_coords, x_coords] = image_matrix[b, n, :, y_coords, x_coords]
                        
        return result_matrix

    def project_points(self, points, world_view_transform, full_proj_transform, fov_x, fov_y, image_width, image_height):
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_homogeneous = torch.cat((points, ones), dim=1)
        points_camera = torch.matmul(points_homogeneous, world_view_transform.T)
        points_image = torch.matmul(points_camera, full_proj_transform.T)
        points_image = points_image / points_image[:, 3].unsqueeze(1)
        points_image[:, 0] = (points_image[:, 0] / torch.tan(fov_x / 2)) * (image_width / 2) + (image_width / 2)
        points_image[:, 1] = (points_image[:, 1] / torch.tan(fov_y / 2)) * (image_height / 2) + (image_height / 2)
        return points_image

    def focal2fov(self, focal, pixels):
        return 2*torch.atan(pixels/(2*focal))


    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape
        imgs, imgs_wo_norm, imgs_wo_aug, c2w = input[8], input[9], input[10], input[11]
        dtype = input[0].dtype

        lss_feat = self.get_lss_bev_feat(input, depth, tran_feat)
        ht_feat, voxel_coor, pseudo_point_cloud = self.get_ht_bev_feat(input, depth, tran_feat)

        [_, pseudo_point_cloud_cam, cam_mask, lidar2img, lidar2img_R, lidar2img_t] = pseudo_point_cloud

        bev_feat = ht_feat
        voxel_feat = self.ObtainVoxelFeature(bev_feat.permute(0, 2, 3, 1).unsqueeze(1))

        B, Height, Width, Length, Channel = voxel_feat.size()

        voxel_coor = voxel_coor.view(B, Height, Width, Length, 3).contiguous()

        cam_mask = cam_mask.view(B, 6, Height, Width, Length, 1).contiguous()
        pseudo_point_cloud_cam = pseudo_point_cloud_cam.view(B, 6, Height, Width, Length, 2).contiguous()

        pseudo_point_cloud_cam = pseudo_point_cloud_cam.view(B, 6, Height, Width * Length, 2).contiguous()
        voxel_coor_for_color = voxel_coor.view(B, Height, Width * Length, 3).contiguous()

        cam_mask = cam_mask.view(B, 6, Height, Width * Length, 1).contiguous()

        pseudo_point_cloud_cam[..., 0] *= self.input_size[1]
        pseudo_point_cloud_cam[..., 1] *= self.input_size[0]

        img_values = self.lidar_points_to_image_values(pseudo_point_cloud_cam, imgs_wo_norm, cam_mask) 

        colored_voxels, _, _ = self.color_voxels(voxel_coor_for_color, img_values, cam_mask, downsample_factor_Z=None, downsample_factor_XY=None)

        colored_voxels = colored_voxels[..., 3:].view(B, Height * Width * Length, 3).contiguous() / 255.0

        voxel_feat = voxel_feat.view(B, Height * Width * Length, Channel).contiguous()
        voxel_coor = voxel_coor.view(B, Height * Width * Length, 3).contiguous()

        imgs_wo_norm_sparse = self.retain_valid_pixels(imgs_wo_norm, pseudo_point_cloud_cam.view(B, 6, Height, Width, Length, 2), cam_mask.view(B, 6, Height, Width, Length, 1))

        camera_intrinsics = input[3]

        cam_idx_list = [random.randint(0, 5) for _ in range(B)]
        gt_images = []
        opacity_list, alpha_list, opacity_alpha_list = [], [], []
        render_image, render_image_G_all, render_image_N_all = [], [], []
        render_depth, render_depth_G_all, render_depth_N_all = [], [], []
        lidar2img_R_clone = lidar2img_R.clone().cpu()
        lidar2img_t_clone = lidar2img_t.clone().cpu()
        camera_intrinsics_clone = camera_intrinsics.clone().cpu()

        for bs in range(B):
            cam_idx = cam_idx_list[bs]

            image_feat_resize_list, sigma_list, alpha_img_list = [], [], []
            for item in range(6):
                image_feat_resize = self.image_feat_resize(input[0][bs][item].unsqueeze(0)).squeeze(0).permute(1, 2, 0).contiguous()
                img_h, img_w, img_ch = image_feat_resize.size()
                image_feat_resize = image_feat_resize.view(1, img_h * img_w, img_ch).contiguous()
                sigma = self.sigma(image_feat_resize)
                image_feat_resize_list.append(image_feat_resize)
                sigma_list.append(sigma)
                raw2alpha = lambda raw: 1.-torch.exp(-raw)
                alpha_img_list.append(raw2alpha(sigma).view(1, img_h, img_w, 1))
                
            image_feat_resize_color = torch.cat((image_feat_resize_list[cam_idx], imgs_wo_norm_sparse[bs][cam_idx].permute(1, 2, 0).view(1, img_h * img_w, 3) / 255.0), -1) 

            color_weight = F.softmax(self.C_MLP_nerf(image_feat_resize_color), dim=-1)
            radiance = self.img_feat_resize1(image_feat_resize_color) * color_weight

            depth_weight = F.softmax(self.D_MLP_nerf(image_feat_resize_color), dim=-1)
            radiance1 = self.img_feat_resize2(image_feat_resize_color) * depth_weight

            raw2alpha = lambda raw: 1.-torch.exp(-raw)
            alpha = raw2alpha(sigma_list[cam_idx])

            T = torch.cumprod(1.-alpha + 1e-10, dim=-1)[..., :-1]
            T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
            weights = alpha * T
            rgb_vr = weights * radiance 
            depth_vr = weights * radiance1
            render_image_N = rgb_vr.reshape(img_h, img_w ,3).permute(2, 0, 1)
            render_depth_N = depth_vr.reshape(img_h, img_w, 1).permute(2, 0, 1)

            alpha_img = torch.cat(alpha_img_list, 0).view(1, 6, 1, img_w, img_h)
            img_values_1 = self.lidar_points_to_image_values(pseudo_point_cloud_cam[bs].unsqueeze(0), alpha_img, cam_mask[bs].unsqueeze(0))
            _, alpha_lidar, _ = self.color_voxels(voxel_coor_for_color[bs].unsqueeze(0), img_values_1, cam_mask[bs].unsqueeze(0), downsample_factor_Z=None, downsample_factor_XY=None)
            alpha_lidar = alpha_lidar.view(1, Height, Width, Length)     
            
            voxel_coor_cv = voxel_coor[bs]

            opacity = self.A_MLP(voxel_feat[bs])
            scaling = self.S_MLP(voxel_feat[bs])
            rotation = self.R_MLP(voxel_feat[bs]) 
            color = self.C_MLP(torch.cat((voxel_feat[bs], colored_voxels[bs]), dim=-1))

            R_ex = lidar2img_R_clone[bs][cam_idx].transpose(0, 1) 
            T_ex = -lidar2img_t_clone[bs][cam_idx].squeeze(1) 

            tar_ixt = camera_intrinsics_clone[bs][cam_idx]
            
            R_ex = c2w[bs][cam_idx][:3, :3].cpu()
            T_ex = c2w[bs][cam_idx][:3, 3].cpu()

            fov_x = self.focal2fov(tar_ixt[0, 0], torch.tensor(self.input_size[1]).float()).cuda()
            fov_y = self.focal2fov(tar_ixt[1, 1], torch.tensor(self.input_size[0]).float()).cuda()

            projection_matrix = torch.tensor(data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt.numpy(), h=self.input_size[0], w=self.input_size[1]).transpose(0, 1)).cuda()
            world_view_transform = torch.tensor(data_utils.getWorld2View2(R_ex.numpy(), T_ex.numpy(), np.array(self.trans), self.scale)).transpose(0, 1).cuda()
            full_proj_transform = torch.tensor((world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)).cuda()
            camera_center = torch.tensor(world_view_transform.inverse()[3, :3]).cuda()

   
            data = {'FovX': fov_x, 'FovY': fov_y, 'height': self.input_size[0], 'width': self.input_size[1], 'world_view_transform': world_view_transform, 'full_proj_transform': full_proj_transform, 'camera_center': camera_center}
            render_image_G, render_depth_G = render(data, cam_idx, voxel_coor_cv, color, rotation, scaling, opacity, bg_color=[0, 0, 0]) # 选择性把远处物体mask掉，只重建远处的物体


            render_image_NG = self.LinearWeightedImage(render_image_G, render_image_N)
            render_depth_NG = self.LinearWeightedDepth(render_depth_G, render_depth_N)

            opacity_up = F.interpolate(opacity.view(1, Height, Width, Length).cuda(), size=(int(Width/6), int(Length/6)), mode='bilinear', align_corners=True)
            alpha_up = F.interpolate(alpha_lidar.cuda(), size=(int(Width/6), int(Length/6)), mode='bilinear', align_corners=True)
            opacity_alpha = F.interpolate(self.defor_cross_attention(opacity_up, alpha_up), size=(Width, Length), mode='bilinear', align_corners=True) + opacity.view(1, Height, Width, Length).cuda()


            render_image.append(render_image_NG.unsqueeze(0))
            render_image_G_all.append(render_image_G.unsqueeze(0))
            render_image_N_all.append(render_image_N.unsqueeze(0))
            render_depth.append(render_depth_NG.unsqueeze(0))
            render_depth_G_all.append(render_depth_G.unsqueeze(0))
            render_depth_N_all.append(render_depth_N.unsqueeze(0))
            gt_images.append(imgs_wo_norm[bs][cam_idx].unsqueeze(0) / 255.0)

            opacity_list.append(opacity)
            alpha_list.append(alpha_lidar)
            opacity_alpha_list.append(opacity_alpha)


        render_image = torch.cat(render_image)
        gt_images = torch.cat(gt_images).cuda()
        render_image_G_all, render_image_N_all = torch.cat(render_image_G_all), torch.cat(render_image_N_all)
        render_depth = torch.cat(render_depth)
        render_depth_G_all, render_depth_N_all = torch.cat(render_depth_G_all), torch.cat(render_depth_N_all)

        channel_feat = self.fuser(lss_feat, ht_feat)
        mask = torch.zeros((B, self.bev_h, self.bev_w),
                    device=ht_feat.device).to(dtype)
        bev_pos = self.positional_encoding(mask).to(dtype)

        bev_mask_logit = self.prob(bev_pos + channel_feat)

        geom_feat = self.geom_att(channel_feat, bev_mask_logit) * channel_feat


        mask1 = torch.zeros((B, self.bev_h, self.bev_w),
                    device=ht_feat.device).to(dtype)
        bev_pos1 = self.positional_encoding1(mask1).to(dtype)
        opacity_alpha_view = self.OpacityVoxelToBEV(torch.cat(opacity_alpha_list, 0).cuda(), bev_pos1)
        opacity_mask = self.ObatinOpacityMask(geom_feat, opacity_alpha_view)
        
        geom_feat = geom_feat * opacity_mask
        
        return geom_feat, depth, bev_mask_logit, [render_image, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all]

    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        gt_semantic_depths = gt_depths * gt_semantics

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                              num_classes=2).view(-1, 2).float()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = (gt_depths -
                     (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:].float()
        gt_semantic_depths = gt_semantic_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantic_depths = gt_semantic_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantic_depths = gt_semantic_depths.view(
            -1, self.downsample * self.downsample)
        gt_semantic_depths =  torch.where(gt_semantic_depths == 0.0,
                                    1e5 * torch.ones_like(gt_semantic_depths),
                                    gt_semantic_depths)
        gt_semantic_depths = (gt_semantic_depths - (self.grid_config['depth'][0] - 
                            self.grid_config['depth'][2])) / self.grid_config['depth'][2] 
        gt_semantic_depths = torch.where(
                    (gt_semantic_depths < self.D + 1) & (gt_semantic_depths >= 0.0),
                    gt_semantic_depths, torch.zeros_like(gt_semantic_depths)).long()                           
        soft_depth_mask = gt_semantics[:,1] > 0
        gt_semantic_depths = gt_semantic_depths[soft_depth_mask]
        gt_semantic_depths_cnt = gt_semantic_depths.new_zeros([gt_semantic_depths.shape[0], self.D+1])
        for i in range(self.D+1):
            gt_semantic_depths_cnt[:,i] = (gt_semantic_depths == i).sum(dim=-1)
        gt_semantic_depths = gt_semantic_depths_cnt[:,1:] / gt_semantic_depths_cnt[:,1:].sum(dim=-1, keepdim=True)
        gt_depths[soft_depth_mask] = gt_semantic_depths

        return gt_depths, gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, depth_labels, depth_preds, semantic_labels, semantic_preds):
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        semantic_weight = torch.zeros_like(semantic_labels[:,1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:,1] > 0] = 0.9

        depth_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]
        semantic_labels = semantic_labels[depth_mask]
        semantic_preds = semantic_preds[depth_mask]
        semantic_weight = semantic_weight[depth_mask]

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ) * semantic_weight).sum() / max(0.1, semantic_weight.sum())

            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_depth_weight * depth_loss, self.loss_semantic_weight * semantic_loss
    
    def depth_l1_loss(self, pred_depth, gt_depth):
        return F.l1_loss(pred_depth, gt_depth)

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        semantic_digit = x[:, self.D:self.D + 2]
        tran_feat = x[:, self.D + 2:self.D + 2 + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        semantic = semantic_digit.softmax(dim=1)
        filter_depth = torch.where(depth < self.depth_threshold, torch.zeros_like(depth), depth)
        img_mask = semantic[:,1:2] >= self.semantic_threshold
        filter_feat = img_mask*tran_feat
        bev_feat, filter_depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all] = self.view_transform(input, filter_depth, filter_feat)

        return bev_feat, depth, (bev_mask, semantic), [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all]


    def Box2dtoMask(self, gt_bboxes, new_width, new_height):

        original_width, original_height = 1600, 900
        scale_x = new_width / original_width

        boxes_resized = gt_bboxes.clone()
        boxes_resized[:, [0, 2]] = gt_bboxes[:, [0, 2]] * scale_x
        boxes_resized[:, [1, 3]] = gt_bboxes[:, [1, 3]] * scale_x

        mask = torch.zeros((1, 3, new_height, new_width), dtype=torch.float32)

        for i in range(boxes_resized.size(0)):
            x1, y1, x2, y2 = boxes_resized[i]
            x1, y1, x2, y2 = int(x1), int(y1) - 158, int(x2), int(y2) - 158
            mask[0, :, y1:y2, x1:x2] = 1.0
        return mask.cuda()

    def get_loss(self, depth, semantic, gt_depth, gt_semantic, gt_bboxes, cam_idx_list, render_imgs, gt_images, render_image_G_all, render_image_N_all, render_depth, render_depth_G_all, render_depth_N_all):
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth, loss_ce_semantic = \
            self.get_depth_and_semantic_loss(depth_labels, depth, semantic_labels, semantic)

        color_loss_list = []
        # perceptual_loss_list = []
        ssim_loss_list = []
        render_depth_loss_list = []

        global iteration
        iteration += 1

        B = render_imgs.shape[0]
        for bs in range(B):
            cam_idx = cam_idx_list[bs]
            gt_bboxes_2d_B = gt_bboxes[bs][cam_idx]
            gt_image_mask = self.Box2dtoMask(gt_bboxes_2d_B, gt_images.shape[3], gt_images.shape[2]) # [1, 3, 256, 704]
            self.visualize_images(gt_images[B].detach().cpu().permute(1, 2, 0).numpy() * 255.0, (gt_images[B] * gt_image_mask.squeeze(0)).detach().cpu().permute(1, 2, 0).numpy() * 255.0)
            
            if iteration > 3862 * 2:
                gt_images_B = gt_images[bs].unsqueeze(0) * gt_image_mask
                render_imgs_B = render_imgs[bs].unsqueeze(0) * gt_image_mask
                render_image_G_B = render_image_G_all[bs].unsqueeze(0) * gt_image_mask
                render_image_N_B = render_image_N_all[bs].unsqueeze(0) * gt_image_mask
            else:
                gt_images_B = gt_images[bs].unsqueeze(0)
                render_imgs_B = render_imgs[bs].unsqueeze(0)
                render_image_G_B = render_image_G_all[bs].unsqueeze(0)
                render_image_N_B = render_image_N_all[bs].unsqueeze(0)

            color_loss_all = self.color_crit(gt_images_B, render_imgs_B)
            color_loss_G = self.color_crit(gt_images_B, render_image_G_B)
            color_loss_N = self.color_crit(gt_images_B, render_image_N_B)

            ssim_all = SSIM(window_size = 11)
            ssim_loss_all = 1 - ssim_all(gt_images_B, render_imgs_B)
            ssim_G = SSIM(window_size = 11)
            ssim_loss_G = 1 - ssim_G(gt_images_B, render_image_G_B)
            ssim_N = SSIM(window_size = 11)
            ssim_loss_N = 1 - ssim_N(gt_images_B, render_image_N_B)

            color_loss_list.append((color_loss_all + color_loss_G + color_loss_N) / 3)
            ssim_loss_list.append((ssim_loss_all + ssim_loss_G + ssim_loss_N) / 3)


            gt_depth_B = gt_depth[bs][cam_idx].unsqueeze(0)
            min_value, max_value = gt_depth_B.min().item(), gt_depth_B.max().item()
            gt_depth_B = (gt_depth_B - min_value) / (max_value - min_value + 0.01)
            render_depth_B = render_depth[bs]
            render_depth_G_B = render_depth_G_all[bs]
            render_depth_N_B = render_depth_N_all[bs]
            render_depth_loss_all = self.depth_l1_loss(render_depth_B, gt_depth_B)
            render_depth_G_loss = self.depth_l1_loss(render_depth_G_B, gt_depth_B)
            render_depth_N_loss = self.depth_l1_loss(render_depth_N_B, gt_depth_B)
            render_depth_loss_list.append((render_depth_loss_all + render_depth_G_loss + render_depth_N_loss) / 3)
            
        color_loss = torch.sum(torch.stack(color_loss_list), dim=0) / B * 20
        ssim_loss = torch.sum(torch.stack(ssim_loss_list), dim=0) / B * 1

        render_depth_loss = torch.sum(torch.stack(render_depth_loss_list) / B * 1)

        return loss_depth, loss_ce_semantic, color_loss, ssim_loss, render_depth_loss