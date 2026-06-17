# Copyright (c) Zhijia. All rights reserved.

# Author: Peidong Li

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
# from .bevdet import BEVDet, BEVDet4D, BEVStereo4D
from .bevdet_vggt import BEVDetVGGT, BEVDet4DVGGT, BEVStereo4DVGGT
from mmdet.models.backbones.resnet import ResNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

@DETECTORS.register_module()
class OcRFDetVGGT(BEVDetVGGT):
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        rots, trans, intrins, post_rots, post_trans, bda, sensor2egos, ego2globals, imgs_wo_norm, imgs_wo_aug, c2w = img[1:]
        mlp_input = self.img_view_transformer.get_mlp_input(
                    rots, trans, intrins, post_rots, post_trans, bda)
        x = self.image_encoder(img[0])
        inputs=[x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, img[0], imgs_wo_norm, imgs_wo_aug, c2w]
        x, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all] = self.img_view_transformer(inputs)
        x = self.bev_encoder(x)
        if self.training:
            return [x], depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all]
        else:
            return [x], depth, bev_mask

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        if self.training:
            img_feats, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts = self.extract_img_feat(img, img_metas, **kwargs)
        else:
            img_feats, depth, bev_mask = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        if self.training:
            return (img_feats, pts_feats, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts)

        else:
            return (img_feats, pts_feats, depth, bev_mask)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        img_metas = img_metas._data[0] ##############################
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all]= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']

        gt_bev_mask = kwargs['gt_bev_mask']

        loss_depth, loss_ce_semantic, loss_gs_color, loss_gs_ssim, loss_render_depth = \
            self.img_view_transformer.get_loss(depth, bev_mask[1], gt_depth, gt_semantic, gt_bboxes, cam_idx_list, render_imgs, gt_images,  render_image_G_all, render_image_N_all, render_depth, render_depth_G_all, render_depth_N_all)
        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic, loss_gs_color=loss_gs_color, loss_gs_ssim=loss_gs_ssim, loss_render_depth=loss_render_depth)

        if opacity_alpha_view != None:
            loss_bev_opacity = self.img_view_transformer.prob.get_bev_opacity_loss(gt_bev_mask, opacity_alpha_view.squeeze(1)) # opacity_alpha_view: [B, 1, 128, 128]
            losses.update(loss_bev_opacity)

        loss_bev_mask = self.img_view_transformer.prob.get_bev_mask_loss(gt_bev_mask, bev_mask[0])
        losses.update(loss_bev_mask)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


def normalize_tensor(x):
    """
    Min-max normalization to [0,1]
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)

@DETECTORS.register_module()
class OcRFDet4DVGGT(OcRFDetVGGT, BEVStereo4DVGGT):

    def resize_img(self, img: torch.Tensor,
                   new_h: int = 294,
                   new_w: int = 518) -> torch.Tensor:
        B, V, C, H, W = img.shape
        img_reshape = img.view(B * V, C, H, W)
        img_resized = F.interpolate(
            img_reshape,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        img_resized = img_resized.view(B, V, C, new_h, new_w)
        return img_resized
    
    
    def get_pair_indices(self):
        i1 = random.randint(0, 5)
        i2 = (i1 + 1) % 6

        special_pairs = {5: 3, 3: 5}
        if i2 in special_pairs:
            i2 = special_pairs[i2]
        return i1, i2
    
    def build_cost_volume(self, feats, max_disp=8, method='concat', groups=32):
        """
        Build stereo cost volume without for-loops.

        Args:
            feats: torch.Tensor, shape [B, 2, C, H, W]
            max_disp: int, max disparity (in feature pixels). D = max_disp + 1
            method: 'concat' | 'group_corr' | 'variance'
            groups: number of groups for group-wise correlation (only for 'group_corr').

        Returns:
            cost_vol: torch.Tensor
        """
        B, V, C, H, W = feats.shape
        assert V == 2, "expected 2-view features for stereo"
        left, right = feats[:, 0], feats[:, 1]  # [B, C, H, W]
        D = max_disp + 1
        device = feats.device

        if method == 'concat':
            idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, D, W)
            shift_idx = torch.clamp(idx - torch.arange(D, device=device).view(1, 1, D, 1), min=0)
            r_shift = right.unsqueeze(2).expand(B, C, D, H, W)
            r_shift = r_shift.permute(0, 1, 3, 2, 4)  # [B, C, H, D, W]
            r_shift = r_shift[..., shift_idx]
            r_shift = r_shift.permute(0, 1, 3, 2, 4)  # [B, C, D, H, W]
            left_expand = left.unsqueeze(2).expand(-1, -1, D, -1, -1)
            cost = torch.cat([left_expand, r_shift], dim=1)  # [B, 2C, D, H, W]
            return cost

        elif method == 'group_corr':
            assert C % groups == 0
            ch_per_group = C // groups
            left_g = left.view(B, groups, ch_per_group, H, W)
            right_g = right.view(B, groups, ch_per_group, H, W)
            W_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, D, W)
            shift_idx = torch.clamp(W_idx - torch.arange(D, device=device).view(1, 1, D, 1), min=0)
            right_g_expand = right_g.unsqueeze(2).expand(B, groups, D, ch_per_group, H, W)  # [B, G, D, Cg, H, W]
            right_g_shifted = torch.gather(right_g_expand, 5, shift_idx.unsqueeze(3).unsqueeze(4).expand(B, groups, D, ch_per_group, H, W))
            left_g_expand = left_g.unsqueeze(2).expand(B, groups, D, ch_per_group, H, W)
            cost = (left_g_expand * right_g_shifted).sum(dim=3)  # [B, groups, D, H, W]
            return cost

        elif method == 'variance':
            left_expand = left.unsqueeze(2).expand(B, C, D, H, W)
            W_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, D, W)
            shift_idx = torch.clamp(W_idx - torch.arange(D, device=device).view(1, 1, D, 1), min=0)
            right_expand = right.unsqueeze(2).expand(B, C, D, H, W)
            right_shifted = torch.gather(right_expand, 4, shift_idx.unsqueeze(1).expand(B, C, D, H, W))
            stacked = torch.stack([left_expand, right_shifted], dim=0)  # [2, B, C, D, H, W]
            cost = torch.var(stacked, dim=0, unbiased=False)
            return cost

        else:
            raise ValueError("Unknown method")


    def prepare_bev_feat(self, img, rot, tran, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame, imgs_wo_norm, imgs_wo_aug, c2w, key_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, None, stereo_feat, None

        stereo = True
        if stereo == True:
            x, stereo_feat = self.image_encoder(img, stereo=stereo) # x: torch.Size([4, 6, 256, 16, 44]), img: torch.Size([4, 6, 3, 256, 704])
            
            if self.training and key_frame:
                x_resize = self.resize_img(img.detach())
                # VGGT, GD
                with torch.no_grad():
                    B, S, _, H, W =  x_resize.shape
                    _, Sf, Cf, Hf, Wf = x.shape
                    cam_idx_list1, cam_idx_list2 = self.get_pair_indices()
                    x_s = torch.cat([
                        x[:, cam_idx_list1:cam_idx_list1+1, ...],
                        x[:, cam_idx_list2:cam_idx_list2+1, ...]
                    ], dim=1)

                    x_t = torch.cat([
                        x_resize[:, cam_idx_list1:cam_idx_list1+1, ...],
                        x_resize[:, cam_idx_list2:cam_idx_list2+1, ...]
                    ], dim=1)
                    aggregated_tokens_list, patch_start_idx = self.aggregator(x_t.cuda())
                    token_img = aggregated_tokens_list[-1][:, :, patch_start_idx:]
                    patch_h, patch_w = H // 14, W // 14
                    token_img = token_img.reshape(B * 2, -1, token_img.shape[-1])
                    token_img = token_img.permute(0, 2, 1).reshape((B, 2, token_img.shape[-1], patch_h, patch_w)) # torch.Size([4, 2, 2048, 21, 37]) 

                x_resize = self.aligner(x_s, token_img)

                cost_volume_t = self.build_cost_volume(token_img, max_disp=4, method='group_corr', groups=64)
                cost_volume_s = self.build_cost_volume(x_resize, max_disp=4, method='group_corr', groups=64)

            metas = dict(k2s_sensor=k2s_sensor,
                        intrins=intrin,
                        post_rots=post_rot,
                        post_trans=post_tran,
                        frustum=self.img_view_transformer.cv_frustum.to(x),
                        cv_downsample=4,
                        downsample=self.img_view_transformer.downsample,
                        grid_config=self.img_view_transformer.grid_config,
                        cv_feat_list=[feat_prev_iv, stereo_feat])
            bev_feat, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all] = self.img_view_transformer(
                [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, img, imgs_wo_norm, imgs_wo_aug, c2w], metas)
        else:
            x, stereo_feat = self.image_encoder(img)
            bev_feat, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all] = self.img_view_transformer(
                [x, rot, tran, intrin, post_rot, post_tran, bda,
                mlp_input, img, imgs_wo_norm, imgs_wo_aug, c2w])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        if self.training and key_frame:
            return bev_feat, depth, bev_mask, stereo_feat, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], [cost_volume_t, cost_volume_s]
        if self.training:
            return bev_feat, depth, bev_mask, stereo_feat, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], None
        else:
            return bev_feat, depth, bev_mask, stereo_feat, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], None


    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda, imgs_wo_norm, imgs_wo_aug, c2w, sensor2keyego, ego2global = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1,
                                                           ...], mlp_input)
        if self.training:
            bev_feat, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts = self.prepare_bev_feat(*inputs_curr)
        else:
            bev_feat, depth, bev_mask= self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        if self.training:
            return [x], depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts
        else:
            return [x], depth, bev_mask

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, **kwargs)
        imgs, rots, trans, intrins, post_rots, post_trans, \
        bda, sensor2keyegos, ego2globals, curr2adjsensor, imgs_wo_norm, imgs_wo_aug, c2ws = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, rot, tran, sensor2keyego, ego2global, intrin, post_rot, post_tran, img_wo_norm, img_wo_aug, c2w = \
                imgs[fid], rots[fid], trans[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid], imgs_wo_norm[fid], imgs_wo_aug[fid], c2ws[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame, img_wo_norm, img_wo_aug, c2w, key_frame)
                if key_frame:
                    bev_feat, depth, bev_mask, feat_curr_iv, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts  = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                    bev_mask_key_frame = bev_mask
                else:
                    with torch.no_grad():
                        bev_feat, depth, bev_mask, feat_curr_iv, _, *others  = \
                            self.prepare_bev_feat(*inputs_curr) # > 3th epoch
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        if self.training:
            return [x], depth_key_frame, bev_mask_key_frame, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts
        else:
            return [x], depth_key_frame, bev_mask_key_frame

    def mean_threshold_iou(self, mask, prediction):
        x, y = np.meshgrid(np.arange(128), np.arange(128))
        depth_map = np.sqrt((x - 64)**2 + (y - 64)**2)
        depth_ranges = [(0, 16), (16, 32), (32, 48), (48, 64)]
        ious = []

        for depth_range in depth_ranges:
            mean_threshold = np.mean(prediction) * 2 / 3
            binary_prediction = prediction >= mean_threshold

            min_depth, max_depth = depth_range
            depth_mask = (depth_map >= min_depth) & (depth_map < max_depth) # (128, 128)
            
            bev_layer = np.logical_and(binary_prediction, depth_mask)
            gt_layer = np.logical_and(mask, depth_mask)
            
            intersection = np.logical_and(gt_layer, bev_layer).sum()
            union = np.logical_or(gt_layer, bev_layer).sum()
            iou = intersection / (union + 0.01)
            ious.append(iou)


        mean_threshold = np.mean(prediction) * 2 / 3
        binary_prediction = prediction >= mean_threshold
        
        intersection = np.logical_and(mask, binary_prediction).sum()
        union = np.logical_or(mask, binary_prediction).sum()
        
        mean_iou = intersection / (union + 0.01)

        
        return ious, mean_iou

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, bev_mask, [render_imgs, gt_images, render_image_G_all, render_image_N_all, opacity_alpha_view, cam_idx_list, render_depth, render_depth_G_all, render_depth_N_all], F_ts = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        

        loss_Smoothl1_KD = F.smooth_l1_loss(F_ts[0], F_ts[1], beta=1.0) * 0.1

        s_flat = F_ts[0].flatten(1)
        t_flat = F_ts[1].flatten(1)
        loss_cos_KD = 1 - F.cosine_similarity(s_flat, t_flat, dim=1).mean()
        
        gt_depth, gt_semantic, gt_bev_mask = kwargs['gt_depth'], kwargs['gt_semantic'], kwargs['gt_bev_mask']
        loss_depth, loss_ce_semantic, loss_gs_color, loss_gs_ssim, loss_render_depth = \
            self.img_view_transformer.get_loss(depth, bev_mask[1], gt_depth, gt_semantic, gt_bboxes, cam_idx_list, render_imgs, gt_images,  render_image_G_all, render_image_N_all, render_depth, render_depth_G_all, render_depth_N_all)
        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic, loss_gs_color=loss_gs_color, loss_gs_ssim=loss_gs_ssim, loss_render_depth=loss_render_depth, loss_Smoothl1_KD=loss_Smoothl1_KD, loss_cos_KD=loss_cos_KD)

        if opacity_alpha_view != None:
            loss_bev_opacity = self.img_view_transformer.prob.get_bev_opacity_loss(gt_bev_mask, opacity_alpha_view.squeeze(1))
            losses.update(loss_bev_opacity)

        loss_bev_mask = self.img_view_transformer.prob.get_bev_mask_loss(gt_bev_mask, bev_mask[0])
        losses.update(loss_bev_mask)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses