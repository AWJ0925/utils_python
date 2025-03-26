model = dict(
    type='MaskRCNN',  # 检测器的名称
    pretrained='torchvision://resnet50',  # ImageNet的预训练模型

    backbone=dict(  # backbone的配置
        type='ResNet',
        # backbone的类型, 请参照 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 查看更多的细节.
        depth=50,  # backbone网络的深度, 对于ResNet and ResNext的backbone而言，通常是使用50或者101的深度
        num_stages=4,  # backbone中stage的个数（应该是相当于ResNet网络中block的个数）
        out_indices=(0, 1, 2, 3),  # backbone中每一个stage过程输出的feature的下标
        frozen_stages=1,  # 1 stage 的权重被冻结
        norm_cfg=dict(  # normalization layers的配置.
            type='BN',  # norm layer的类型, 通常是 BN or GN
            requires_grad=True),  # 是否训练BN中的gamma and beta参数
        norm_eval=True,  # 是否冻结BN中的统计信息（相当于模型eval的过程，不进行统计数据）
        style='pytorch'),
    # backbone的类型, 'pytorch' means that stride 2 layers are in 3x3 conv,
    # 'caffe' means stride 2 layers are in 1x1 convs.(感觉这句直接看英文还方便些)

    neck=dict(  # neck模块的配置
        type='FPN',
        # 该detection的neck为FPN. 我们还提供了 'NASFPN', 'PAFPN'等neck类型. 具体请参照 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py#L10 查看更多的细节。
        in_channels=[256, 512, 1024, 2048],  # 输入的channels,这个地方和backbone的output channels保持一直。
        out_channels=256,  # 特征金字塔(pyramid feature map)的每一层输出的channel数
        num_outs=5),  # output 输出的个数

    rpn_head=dict(  # RPN模块的配置
        type='RPNHead',
        # RPN head 的类型为'RPNHead', 我们还支持 'GARPNHead'等等. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12.
        in_channels=256,  # 每一个输入的feature的input channels, 这个地方需要和neck模块的output channels保持一致。
        feat_channels=256,
        # Feature channels of convolutional layers in the head
        # （应该是指RPN模块头部的卷积操作，输出channel为256，它的输入为上面FPN得到的多尺度feature map）.
        anchor_generator=dict(  # 生成anchor的配置
            type='AnchorGenerator',
            # 绝大多数都是用AnchorGenerator, SSD 检测器(单阶段的目标检测算法)使用的是`SSDAnchorGenerator`. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10.
            scales=[8],  # anchor的生成个数, 特征图上每一个位置所生成的anchor个数为scale * base_sizes
            ratios=[0.5, 1.0, 2.0],  # anchor中height 和width的比率.
            strides=[4, 8, 16, 32, 64]),
        # The strides of the anchor generator. 这个需要和FPN feature strides保持一致. 如果base_sizes没有设置的话，这个strides 将会被当作base_sizes.
        bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
            type='DeltaXYWHBBoxCoder',
            # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9 for more details.
            target_means=[0.0, 0.0, 0.0, 0.0],  # The target means used to encode and decode boxes
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # The standard variance used to encode and decode boxes
        loss_cls=dict(  # 分类分支的损失函数配置
            type='CrossEntropyLoss',  # 分类分支的损失函数类型, 我们也提供FocalLoss等损失函数
            use_sigmoid=True,  # RPN 过程通常是一个二分类，所以它通常使用sigmoid函数。
            loss_weight=1.0),  # 分类损失分支所占的权重。
        loss_bbox=dict(  # box回归分支的损失函数配置.
            type='L1Loss',
            # loss的类型, 我们还提供了许多IoU Losses and smooth L1-loss 等. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56.
            loss_weight=1.0)),  # 回归分支损失所占的权重.

    roi_head=dict(  # RoIHead 封装了二阶段检测器的第二阶段的模块
        type='StandardRoIHead',
        # RoI head的类型. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10.
        bbox_roi_extractor=dict(  # RoI feature extractor 用于 bbox regression.
            type='SingleRoIExtractor',
            # RoI feature extractor的类型, 绝大多少方法都使用 SingleRoIExtractor. 具体实现细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10.
            roi_layer=dict(  # RoI Layer的配置
                type='RoIAlign',
                # RoI Layer的类型, 同时还支持DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack这两种类型. 具体实现细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79.
                output_size=7,  # feature maps的输出尺度，相当于输出7*7.
                sampling_ratio=0),
            # Sampling ratio when extracting the RoI features. 0 means adaptive ratio.（这个参数我还不太明白orz）
            out_channels=256,  # 提取特征的输出channels数.
            featmap_strides=[4, 8, 16, 32]),
        # Strides of multi-scale feature maps. It should be consistent to the architecture of the backbone.（这个地方还不太清楚）
        bbox_head=dict(  # RoIHead中的 bbox head的配置.
            type='Shared2FCBBoxHead',
            # bbox head的类型, 具体细节请参照 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177.
            in_channels=256,  # bbox head的输入channels数. 这个地方需要和roi_extractor的out_channels保持一致。
            fc_out_channels=1024,  # FC layers的输出维度.
            roi_feat_size=7,  # RoI features的尺寸
            num_classes=80,  # 分类类别数
            bbox_coder=dict(  # Box coder used in the second stage.
                type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods.
                target_means=[0.0, 0.0, 0.0, 0.0],  # Means used to encode and decode box
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
            reg_class_agnostic=False,  # Whether the regression is class agnostic.
            loss_cls=dict(  # 分类分支的损失函数配置
                type='CrossEntropyLoss',  # 分类分支损失函数的类型, 我们还提供了FocalLoss 等.
                use_sigmoid=False,  # 是否使用sigmoid.
                loss_weight=1.0),  # 分类分支损失所占的权重.
            loss_bbox=dict(  # 回归分支损失函数配置.
                type='L1Loss',  # 损失函数类型, 我们还提供了许多IoU Losses和smooth L1-loss等.
                loss_weight=1.0)),  # 回归分支损失所占的权重.

        mask_roi_extractor=dict(  # RoI feature extractor 用于 mask regression.
            type='SingleRoIExtractor',  # RoI feature extractor的类型, 绝大多数方法都是使用SingleRoIExtractor.
            roi_layer=dict(  # RoI Layer 的配置，提取特征用于实例分割。
                type='RoIAlign',  # RoI Layer的类型,我们还提供了DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack.
                output_size=14,  # feature maps的输出size.
                sampling_ratio=0),  # Sampling ratio when extracting the RoI features.（这个参数还没太弄明白）
            out_channels=256,  # extracted feature的输出channels.
            featmap_strides=[4, 8, 16, 32]),  # Strides of multi-scale feature maps.(这个参数没太弄明白)
        mask_head=dict(  # Mask 的预测模块
            type='FCNMaskHead',
            # mask head的类型, 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L21.
            num_convs=4,  # mask head中卷积层的个数.
            in_channels=256,  # mask head输入的channels数, 应该和mask roi extractor的输出channel数保持一致。
            conv_out_channels=256,  # convolutional layer输出的channel数.
            num_classes=80,  # 分割任务的类别数
            loss_mask=dict(  # mask 分支的损失函数配置.
                type='CrossEntropyLoss',  # 用于分割的损失函数类型
                use_mask=True,  # Whether to only train the mask in the correct class（是否训练仅仅是正确类别的mask）.
                loss_weight=1.0))))  # mask分支损失所占的权重.

train_cfg = dict(  # 训练过程中rpn and rcnn和模块的超参数设置
    rpn=dict(  # 训练过程中rpn的超参数配置
        assigner=dict(  # assigner的配置（assigner是个什么东西？可以理解为一个超参配置的字典吧）
            type='MaxIoUAssigner',
            # assigner的类型, MaxIoUAssigner被用在许多常见的detectors. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10.
            pos_iou_thr=0.7,  # IoU >= threshold 0.7 将会被当作一个正样本
            neg_iou_thr=0.3,  # IoU < threshold 0.3 将会被当作一个负样本
            min_pos_iou=0.3,  # The minimal IoU threshold to take boxes as positive samples
            match_low_quality=True,  # Whether to match the boxes under low quality (see API doc for more details).
            ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
        sampler=dict(  # positive/negative sampler的配置
            type='RandomSampler',
            # sampler的类型, 同时还提供有PseudoSampler和其他类型的samplers.具体实现细节请参照Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8.
            num=256,  # samples的个数
            pos_fraction=0.5,  # 正样本占总样本的比例。
            neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
            add_gt_as_proposals=False),  # Whether add GT as proposals after sampling.
        allowed_border=-1,  # The border allowed after padding for valid anchors.
        pos_weight=-1,  # The weight of positive samples during training.
        debug=False),  # 是否设置debug 模式
    rpn_proposal=dict(  # 在训练过程中生成proposals的配置
        nms_across_levels=False,  # Whether to do NMS for boxes across levels
        nms_pre=2000,  # 在NMS之前的box个数
        nms_post=1000,  # NMS处理后保留的box个数
        max_num=1000,  # NMS处理之后所使用的box个数
        nms_thr=0.7,  # NMS过程所使用的阈值
        min_bbox_size=0),  # 允许的最小的box尺寸

    rcnn=dict(  # roi heads的超参数配置
        assigner=dict(  # 第二阶段的assigner配置, 这个和上面rpn中用到的assigner有所不同
            type='MaxIoUAssigner',
            # assigner的类型, MaxIoUAssigner被用于所有的roi_heads. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10.
            pos_iou_thr=0.5,  # IoU >= threshold 0.5 被当作正样本
            neg_iou_thr=0.5,  # IoU >= threshold 0.5 被当作正样本
            min_pos_iou=0.5,  # 最小的IoU 阈值来判断 boxes 是否为正样本。
            match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
            ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
        sampler=dict(
            type='RandomSampler',
            # sampler的类型, 还提供PseudoSampler和其他的samplers类型. 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8.
            num=512,  # 样例的个数
            pos_fraction=0.25,  # 正样例占总样例的比例。
            neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
            add_gt_as_proposals=True
        ),  # 在sample过程之后，是否将ground trueth当作proposals.
        mask_size=28,  # mask的size大小
        pos_weight=-1,  # The weight of positive samples during training(不太明白).
        debug=False))  # 是否设置debug mode

test_cfg = dict(  # rpn and rcnn在测试过程的超参数配置
    rpn=dict(  # 在测试过程rpn生成proposals的配置（相当于第一阶段）
        nms_across_levels=False,  # Whether to do NMS for boxes across levels
        nms_pre=1000,  # NMS之前的boxs个数
        nms_post=1000,  # NMS所保留的boxs个数
        max_num=1000,  # NMS处理之后最多被使用的boxs个数
        nms_thr=0.7,  # 在NMS处理过程中所使用的阈值
        min_bbox_size=0),  # 允许的最小的box尺寸
    rcnn=dict(  # roi heads的配置
        score_thr=0.05,  # 用来过滤boxes的阈值
        nms=dict(  # nms 在第二阶段的配置
            type='nms',  # nms的类型
            iou_thr=0.5),  # NMS的阈值
        max_per_img=100,  # Max number of detections of each image
        mask_thr_binary=0.5))  # mask 预测的阈值

dataset_type = 'CocoDataset'  # Dataset的类型, 将用于定义数据集
data_root = 'data/coco/'  # 数据集的存放路径
img_norm_cfg = dict(  # 对输入图片进行标准化处理的配置
    mean=[123.675, 116.28, 103.53],  # 用于预训练backbone模型的均值
    std=[58.395, 57.12, 57.375],  # 用于预训练backbone模型的标准差
    to_rgb=True
)  # The channel orders of image used to pre-training the pre-trained backbone models
