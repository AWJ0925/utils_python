optimizer = dict(  # 构造optimizer的配置, 支持PyTorch中所有的优化器，并且参数名称也和PyTorch中提供的一样。
    type='SGD',
    # optimizers的类型, 具体细节请参照https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13.
    lr=0.02,  # optimizers的学习率, 请到PyTorch的文档中查看相关参数的具体用法。
    momentum=0.9,  # SGD优化器的超参数：Momentum
    weight_decay=0.0001)  # SGD优化器的超参数：Weight decay
optimizer_config = dict(
    # 构造optimizer hook的配置, 具体细节请参照 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8.
    grad_clip=None)  # 绝大多少方法都不会使用gradient clip
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',
    # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # warmup的策略, 还支持 `exp` 和 `constant`.
    warmup_iters=500,  # warmup的迭代次数
    warmup_ratio=0.001,  # 用于warmup的起始学习比率
    step=[8, 11])  # 学习率进行衰减的step位置

total_epochs = 12  # model训练的总epoch数

checkpoint_config = dict(
    # 设置checkpoint hook, 具体细节请参照https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py 的实现.
    interval=1)  # 每隔几个epoch保存一下checkpoint

log_config = dict(  # logger文件的配置
    interval=50,  # 每隔多少个epoch输出一个log文件
    hooks=[
        # dict(type='TensorboardLoggerHook')  # MMDetection支持Tensorboard logger
        dict(type='TextLoggerHook')
    ])  # logger 被用来记录训练过程.

dist_params = dict(backend='nccl')  # 设置分布式训练的参数，也可以设置端口。
log_level = 'INFO'  # The level of logging.
load_from = None  # 给出之前预训练模型checkpoint的路径，这个不会resume training（resume training会按照上次的记录接着训练，而这个参数应该只是导入之前预训练模型参数，重新训练）
resume_from = None  # 给出需要Resume 的checkpoints的路径, 它将会接着上次被保存的地方进行训练。
workflow = [('train',
             1)]
# Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.（这个workflow具体是干什么的我不是很清楚orz）
work_dir = 'work_dir'  # 保存模型的文件夹路径（checkpoints和log文件都会保存在其中）。
