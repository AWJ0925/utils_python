train_pipeline = [  # 训练的pipeline
    dict(type='LoadImageFromFile'),  # First pipeline用于从文件存放路径中导入图片
    dict(
        type='LoadAnnotations',  # Second pipeline用于给图片导入对应的标签
        with_bbox=True,  # 是否使用bounding box标签数据, 如果用于检测任务，则为True
        with_mask=True,  # 是否使用instance mask标签数据, 如果用于实例分割任务，则为True
        poly2mask=False),  # 是否将polygon mask转化为instance mask, 设置为False将会加速和减少内存
    dict(
        type='Resize',  # Augmentation pipeline resize图片和图片所对应的标签
        img_scale=(1333, 800),  # 图片的最大尺寸
        keep_ratio=True  # 是否保存宽高比例
    ),
    dict(
        type='RandomFlip',  # Augmentation pipeline flip图片和图片所对应的标签
        flip_ratio=0.5),  # flip的比率
    dict(
        type='Normalize',  # Augmentation pipeline 对输入的图片进行标准化
        mean=[123.675, 116.28, 103.53],  # 均值
        std=[58.395, 57.12, 57.375],  # 标准差
        to_rgb=True),
    dict(
        type='Pad',  # Padding 的配置
        size_divisor=32),  # 填充图像的数目应该可以被整除
    dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
    dict(
        type='Collect',  # 决定数据中哪些key可以被传入pipeline中
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline 从文件路径中导入图片
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1333, 800),  # 用于Resize pipeline的最大图片尺寸
        flip=False,  # 是否在test过程flip images
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                 keep_ratio=True),  # 是否保持宽高的比例.
            dict(type='RandomFlip'),  # 由于flip=False这个RandomFlio将不会被使用。
            dict(
                type='Normalize',  # 标准化操作的配置, 从img_norm_cfg文件中取相应的值
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='Pad',  # padding图片使其能够被12整除.
                size_divisor=32),
            dict(
                type='ImageToTensor',  # 将图片转化为tensor
                keys=['img']),
            dict(
                type='Collect',  # Collect pipeline 收集在test过程中必要的key.
                keys=['img'])
        ])
]

data = dict(
    # 学习率lr和总的batch size数目成正比，例如：8卡GPU  samples_per_gpu = 2的情况（相当于总的batch size = 8*2）,学习率lr = 0.02
    # 如果我是单卡GPU samples_per_gpu = 4的情况，学习率lr应该设置为:0.02*(4/16) = 0.005
    samples_per_gpu=2,  # 每个GPU上的batch size
    workers_per_gpu=2,  # 每个GPU上的workers数目
    train=dict(  # 训练数据集的配置
        type='CocoDataset',
        # 数据集的类型, 具体信息请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19.
        ann_file='data/coco/annotations/instances_train2017.json',  # 标注文件的路径
        img_prefix='data/coco/train2017/',  # 图片文件的前缀
        pipeline=[
            # pipeline, this is passed by the train_pipeline created before.（这个地方应该可以直接写成pipeline = train_pipeline,因为上面有定义train_pipeline这个中间变量）
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(  # 验证集的配置
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(  # Test dataset config, modify the ann_file for test-dev/test submission
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        samples_per_gpu=2  # 测试过程中每张GPU上的batch size
    ))
evaluation = dict(
    # 这个配置是创建一个evaluation hook, 具体细节请查看https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7.
    interval=1,  # 隔多少个epoch进行evaluation一次
    metric=['bbox', 'segm'])  # evaluation所用的评价指标