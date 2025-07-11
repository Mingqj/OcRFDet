_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='OcRFDet4D',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='OcRFViewTransformerFull',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.
                          ),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.3, 0.1, 0.1, 0.1, 0.1, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        load_point_label=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='GetBEVMask', point_cloud_range=point_cloud_range, voxel_size=[0.8,0.8,8.], downsample_ratio=1.),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth', 'gt_semantic', 'gt_bev_mask'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root+"bevdetv2-nuscenes_infos_train.pkl",
    )

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root+"bevdetv2-nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
lr_all = 2e-4
lr_mult_all = 2
optimizer = dict(type='AdamW', lr=lr_all, weight_decay=1e-2, 
                 paramwise_cfg=dict(custom_keys=
                                    {'img_view_transformer.S_MLP': dict(lr_mult=lr_mult_all), 
                                     'img_view_transformer.R_MLP': dict(lr_mult=lr_mult_all),
                                     'img_view_transformer.A_MLP': dict(lr_mult=lr_mult_all),
                                     'img_view_transformer.C_MLP': dict(lr_mult=lr_mult_all),
                                     'img_view_transformer.C_MLP_nerf': dict(lr_mult=lr_mult_all),
                                     'img_view_transformer.D_MLP_nerf': dict(lr_mult=lr_mult_all),
                                     'img_view_transformer.sigma': dict(lr_mult=lr_mult_all),
                                     'img_feat_resize1': dict(lr_mult=lr_mult_all),
                                     'img_feat_resize2': dict(lr_mult=lr_mult_all),
                                     }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=20)

custom_hooks = [
    dict(type='CustomLrUpdaterHook', layers=[
        'img_view_transformer.S_MLP.fc1.weight',
        'img_view_transformer.S_MLP.fc1.bias', 
        'img_view_transformer.S_MLP.fc2.weight',
        'img_view_transformer.S_MLP.fc2.bias', 

        'img_view_transformer.R_MLP.fc1.weight',
        'img_view_transformer.R_MLP.fc1.bias', 
        'img_view_transformer.R_MLP.fc2.weight',
        'img_view_transformer.R_MLP.fc2.bias', 

        'img_view_transformer.A_MLP.fc1.weight',
        'img_view_transformer.A_MLP.fc1.bias', 
        'img_view_transformer.A_MLP.fc2.weight',
        'img_view_transformer.A_MLP.fc2.bias', 

        'img_view_transformer.C_MLP.fc1.weight',
        'img_view_transformer.C_MLP.fc1.bias', 
        'img_view_transformer.C_MLP.fc2.weight',
        'img_view_transformer.C_MLP.fc2.bias', 

        'img_view_transformer.C_MLP_nerf.fc1.weight',
        'img_view_transformer.C_MLP_nerf.fc1.bias',
        'img_view_transformer.C_MLP_nerf.fc2.weight',
        'img_view_transformer.C_MLP_nerf.fc2.bias',

        'img_view_transformer.D_MLP_nerf.fc1.weight',
        'img_view_transformer.D_MLP_nerf.fc1.bias',
        'img_view_transformer.D_MLP_nerf.fc2.weight',
        'img_view_transformer.D_MLP_nerf.fc2.bias',

        'img_view_transformer.img_feat_resize1.fc1.weight',
        'img_view_transformer.img_feat_resize1.fc1.bias',
        'img_view_transformer.img_feat_resize1.fc2.weight',
        'img_view_transformer.img_feat_resize1.fc2.bias',
        'img_view_transformer.img_feat_resize2.fc1.weight',
        'img_view_transformer.img_feat_resize2.fc1.bias',
        'img_view_transformer.img_feat_resize2.fc2.weight',
        'img_view_transformer.img_feat_resize2.fc2.bias',
        
        'img_view_transformer.sigma.0',
        'img_view_transformer.sigma.1',
    ], step=1, gamma=1, priority='VERY_LOW'),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2
    )
]