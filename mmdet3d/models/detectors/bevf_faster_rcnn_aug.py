import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn


class SE_Block1(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=2, dropout_rate=0.1):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.dropout = nn.Dropout(dropout_rate)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_key_value):
        b, n, c = query_key_value.size()  # [B, N, C]

        qkv = self.norm(query_key_value)

        qkv = self.qkv(qkv)  # [B, N, 3C]
        qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)  # [B, N, num_heads, c // self.num_heads]

        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.scale
        attention = scores.softmax(dim=-1)
        attention = self.dropout(attention)
        # Combine heads
        attended = torch.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().reshape(b, n, c)
        # Project back to the original size and reshape
        attended = self.out_projection(attended)
        attended = self.dropout(attended)

        attended = query_key_value + attended

        attended_norm = self.norm(attended)
        mlp_output = self.mlp(attended_norm)
        output = attended + mlp_output

        return output


class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=2, dropout_rate=0.1):
        super(MultiHeadCrossAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.dropout = nn.Dropout(dropout_rate)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value):
        b, n, c = query.size()  # [B, N, C]

        q = self.norm(query)
        k_v = self.norm(key_value)

        q = self.q(q)  # [B, N, C]
        k_v = self.kv(k_v)  # [B, N, 2C]

        q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k_v = k_v.reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = k_v.unbind(0)  # [B, N, num_heads, c // self.num_heads]

        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.scale
        attention = scores.softmax(dim=-1)
        attention = self.dropout(attention)
        # Combine heads
        attended = torch.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().reshape(b, n, c)
        # Project back to the original size and reshape
        attended = self.out_projection(attended)
        attended = self.dropout(attended)

        attended = key_value + attended

        attended_norm = self.norm(attended)
        mlp_output = self.mlp(attended_norm)
        output = attended + mlp_output

        return output


class SequentialCrossAttention(nn.Module):
    def __init__(self, embed_dim1=256, embed_dim2=512, drop_path_ratio=0.1):
        super(SequentialCrossAttention, self).__init__()

        self.mhca = MultiHeadCrossAttentionBlock(dropout_rate=drop_path_ratio)
        self.mhsa = MultiHeadSelfAttentionBlock(dropout_rate=drop_path_ratio)
        self.pos_embed = nn.Parameter(torch.randn(1, 8100, 256))

        self.norm1 = nn.LayerNorm(embed_dim1)
        self.norm2 = nn.LayerNorm(embed_dim2)
        self.dropout = nn.Dropout(drop_path_ratio)

        self.cc1 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.cc2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.cc3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.cc4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.cc5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm3 = nn.SyncBatchNorm(512)
        self.act = nn.ReLU()
        self.weight1 = nn.Parameter(torch.zeros(1))

    def forward(self, feature_a, feature_b, pts_feats_change_):
        batch_size, c, h, w = feature_a.size()
        C = 2 * c

        feature_fusion1 = torch.cat([feature_a, pts_feats_change_], dim=1)
        feature_fusion1_ = self.cc1(feature_fusion1)
        feature_fusion2 = feature_fusion1_.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        feature_fusion2 = self.norm2(feature_fusion2)
        feature_fusion2 = self.act(feature_fusion2)  # [b,n,512]
        feature_fusion2_ = feature_fusion2.transpose(1, 2).contiguous().reshape(batch_size, C, h, w)

        feature_fusion_list = []

        feature_fusion3_ = self.cc2(feature_fusion2_)  # [b, 512, 90, 90]
        feature_fusion_list.append(feature_fusion3_)

        feature_fusion4_ = self.cc3(feature_fusion3_)  # [b, 256, 90, 90]
        feature_fusion4 = feature_fusion4_.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        feature_fusion_list.append(feature_fusion4)

        feature_fusion5_ = self.cc4(feature_fusion4_)  # [b, 256, 90, 90]
        feature_fusion5 = feature_fusion5_.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        feature_fusion_list.append(feature_fusion5)

        feature_a = feature_a.flatten(2).transpose(1, 2).contiguous()  # [B, N, C] 256
        feature_a = self.norm1(feature_a)
        feature_b = feature_b.flatten(2).transpose(1, 2).contiguous()  # [B, N, C] 256
        feature_b = self.norm1(feature_b)

        pos_embed = self.pos_embed

        feature_a = feature_a + pos_embed
        feature_a = self.dropout(feature_a)
        feature_b = feature_b + pos_embed
        feature_b = self.dropout(feature_b)

        feature_c = self.mhca(feature_b, feature_a)  # Module A
        feature_d = self.mhca(feature_c, feature_b)  # Module B
        feature_c = feature_c + feature_fusion_list[1]
        feature_d = feature_d + feature_fusion_list[1]
        feature_b, feature_a = feature_d, feature_c

        feature_c = self.mhca(feature_b, feature_a)  # Module A
        feature_d = self.mhca(feature_c, feature_b)  # Module B
        feature_c = feature_c + feature_fusion_list[2]
        feature_d = feature_d + feature_fusion_list[2]
        feature_b, feature_a = feature_d, feature_c

        feature_fuse = torch.cat([feature_b, feature_a], dim=2)  # [B, N, C] 512
        feature_fuse = self.mhsa(feature_fuse)
        feature_fuse = self.norm2(feature_fuse)
        feature_fuse = feature_fuse.transpose(1, 2).contiguous().reshape(batch_size, C, h, w)
        weight1 = torch.sigmoid(self.weight1)
        # print("weight1:", weight1)
        feature_fuse = weight1 * feature_fuse + (1 - weight1) * feature_fusion_list[0]
        feature_fuse = self.cc5(feature_fuse)
        feature_fuse = self.norm3(feature_fuse)
        feature_fuse = self.act(feature_fuse)

        return feature_fuse

# class SE_Block(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, kernel_size=1, stride=1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return x * self.att(x)


@DETECTORS.register_module()
class BEVF_FasterRCNN_Aug(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False,
                final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, imc=256, lic=384, **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(BEVF_FasterRCNN_Aug, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        # self.se = se

        self.cc1 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.cc2 = nn.Conv2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.cc3 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=True)

        self.sequential_attention = SequentialCrossAttention()

        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)

        self.seblock1 = SE_Block1()

        # if lc_fusion:
        #     if se:
        #         self.seblock = SE_Block(lic)
        #     self.reduc_conv = ConvModule(
        #        lic + imc,
        #        lic,
        #        3,
        #        padding=1,
        #        conv_cfg=None,
        #        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        #        act_cfg=dict(type='ReLU'),
        #        inplace=False)

        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.requires_grad = False

    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    def extract_feat(self, points, img, img_metas, img_aug_matrix=None, lidar_aug_matrix=None, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN//self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:  
                    mat = torch.Tensor(mat).to(img_feats_view.device)
                    rot_list.append(mat.inverse()[:3, :3])
                    trans_list.append(mat.inverse()[:3, 3].view(-1))

                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']

            post_rots = None
            post_trans = None
            if img_aug_matrix is not None:
                img_aug_matrix = torch.stack(img_aug_matrix).permute(1, 0, 2, 3)
                post_rots = img_aug_matrix[..., :3, :3]
                post_trans = img_aug_matrix[..., :3, 3]
            
            extra_rots = None
            extra_trans = None
            if lidar_aug_matrix is not None:
                lidar_aug_matrix = lidar_aug_matrix.unsqueeze(1).repeat(1, self.num_views, 1, 1)
                extra_rots = lidar_aug_matrix[..., :3, :3]
                extra_trans = lidar_aug_matrix[..., :3, 3]
            img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas,
                                                                post_rots=post_rots, post_trans=post_trans, extra_rots=extra_rots,extra_trans=extra_trans)

            if pts_feats is None:
                pts_feats = [img_bev_feat]  # cam stream only!!!
            else:
                if self.lc_fusion:
                    # pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    # if self.se:
                    #     pts_feats = [self.seblock(pts_feats[0])]

                    img_bev_feat_change = self.cc1(img_bev_feat)  # 90x90_256_position_embedding_4layers_2heads
                    pts_feats_change = self.cc2(pts_feats[0])
                    pts_feats_change_ = self.cc3(pts_feats[0])
                    output_b = self.sequential_attention(img_bev_feat_change, pts_feats_change, pts_feats_change_)
                    output = self.seblock1(output_b)
                    output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                    pts_feats = [output]
        return dict(
            img_feats=img_feats,
            pts_feats=pts_feats,
            depth_dist=depth_dist
        )
    
    def simple_test(self, points, img_metas, img=None, img_aug_matrix=None, rescale=False):
        """Test function without augmentaiton."""
        if img_aug_matrix is not None:
            img_aug_matrix = img_aug_matrix[0]
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, img_aug_matrix=img_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      img_aug_matrix=None,
                      lidar_aug_matrix=None,
                      proposals=None,
                      gt_bboxes_ignore=None):


        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, img_aug_matrix=img_aug_matrix, lidar_aug_matrix=lidar_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses
    
    def depth_dist_loss(self, predict_depth_dist, gt_depth, loss_method='kld', img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
        mask = (min_depth>=self.camera_depth_range[0]) & (min_depth<=self.camera_depth_range[1])
        mask = mask.view(-1)
        guassian_depth = guassian_depth.view(-1, D)[mask]
        predict_depth_dist = predict_depth_dist.permute(0, 1, 3, 4, 2).reshape(-1, D)[mask]
        if loss_method=='kld':
            loss = F.kl_div(torch.log(predict_depth_dist), guassian_depth, reduction='mean', log_target=False)
        elif loss_method=='mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss
