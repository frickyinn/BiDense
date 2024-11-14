import argparse
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import thop.vision

from configs.segmentation.default import get_cfg_defaults
from misc import seed_torch
from datasets.segmentation_dataloader import get_dataLoader
from pl_trainer_segmentation import PL_SegmentationTrainer


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    seed = config.RANDOM_SEED
    seed_torch(seed)

    dataloader_args = dict(
        data_path=config.DATASET.DATA_PATH,
        base_size=config.DATASET.BASE_SIZE,
        crop_size=config.DATASET.CROP_SIZE,
        batch_size=2,
        num_threads=config.TRAINING.NUM_THREADS,
    )
    train_dataloader = get_dataLoader(dataset=config.DATASET.DATASET, mode='train', **dataloader_args)
    valid_dataloader = get_dataLoader(dataset=config.DATASET.DATASET, mode='val', **dataloader_args)
    
    trainer_args = dict(
        model_type=config.MODEL.MODEL_TYPE,
        binary_type=config.MODEL.BINARY_TYPE,
        dataset=config.DATASET.DATASET,
        max_lr=config.TRAINING.MAX_LR,
        epochs=config.TRAINING.NUM_EPOCHS,
        num_classes=config.MODEL.NUM_CLASSES,
    )
    if args.weights is None:
        segmentation_trainer = PL_SegmentationTrainer(**trainer_args)
    else:
        segmentation_trainer = PL_SegmentationTrainer.load_from_checkpoint(checkpoint_path=args.weights, **trainer_args)

    # parameters = [p.numel() for p in segmentation_trainer.module.parameters()]
    # parameters = []
    state_dict = segmentation_trainer.module.state_dict()
    keys = [k for k in state_dict]
    bi_keys = [
        'backbone.downsample_layers.1.conv.conv.weight', 'backbone.downsample_layers.2.conv.conv.weight', 'backbone.downsample_layers.3.conv.conv.weight', 'backbone.stages.0.0.dwconv.conv.weight', 'backbone.stages.0.0.pwconv1.conv.weight', 'backbone.stages.0.0.pwconv2.conv.weight', 'backbone.stages.0.1.dwconv.conv.weight', 'backbone.stages.0.1.pwconv1.conv.weight', 'backbone.stages.0.1.pwconv2.conv.weight', 'backbone.stages.0.2.dwconv.conv.weight', 'backbone.stages.0.2.pwconv1.conv.weight', 'backbone.stages.0.2.pwconv2.conv.weight', 'backbone.stages.1.0.dwconv.conv.weight', 'backbone.stages.1.0.pwconv1.conv.weight', 'backbone.stages.1.0.pwconv2.conv.weight', 'backbone.stages.1.1.dwconv.conv.weight', 'backbone.stages.1.1.pwconv1.conv.weight', 'backbone.stages.1.1.pwconv2.conv.weight', 'backbone.stages.1.2.dwconv.conv.weight', 'backbone.stages.1.2.pwconv1.conv.weight', 'backbone.stages.1.2.pwconv2.conv.weight', 'backbone.stages.2.0.dwconv.conv.weight', 'backbone.stages.2.0.pwconv1.conv.weight', 'backbone.stages.2.0.pwconv2.conv.weight', 'backbone.stages.2.1.dwconv.conv.weight', 'backbone.stages.2.1.pwconv1.conv.weight', 'backbone.stages.2.1.pwconv2.conv.weight', 'backbone.stages.2.2.dwconv.conv.weight', 'backbone.stages.2.2.pwconv1.conv.weight', 'backbone.stages.2.2.pwconv2.conv.weight', 'backbone.stages.2.3.dwconv.conv.weight', 'backbone.stages.2.3.pwconv1.conv.weight', 'backbone.stages.2.3.pwconv2.conv.weight', 'backbone.stages.2.4.dwconv.conv.weight', 'backbone.stages.2.4.pwconv1.conv.weight', 'backbone.stages.2.4.pwconv2.conv.weight', 'backbone.stages.2.5.dwconv.conv.weight', 'backbone.stages.2.5.pwconv1.conv.weight', 'backbone.stages.2.5.pwconv2.conv.weight', 'backbone.stages.2.6.dwconv.conv.weight', 'backbone.stages.2.6.pwconv1.conv.weight', 'backbone.stages.2.6.pwconv2.conv.weight', 'backbone.stages.2.7.dwconv.conv.weight', 'backbone.stages.2.7.pwconv1.conv.weight', 'backbone.stages.2.7.pwconv2.conv.weight', 'backbone.stages.2.8.dwconv.conv.weight', 'backbone.stages.2.8.pwconv1.conv.weight', 'backbone.stages.2.8.pwconv2.conv.weight', 'backbone.stages.3.0.dwconv.conv.weight', 'backbone.stages.3.0.pwconv1.conv.weight', 'backbone.stages.3.0.pwconv2.conv.weight', 'backbone.stages.3.1.dwconv.conv.weight', 'backbone.stages.3.1.pwconv1.conv.weight', 'backbone.stages.3.1.pwconv2.conv.weight', 'backbone.stages.3.2.dwconv.conv.weight', 'backbone.stages.3.2.pwconv1.conv.weight', 'backbone.stages.3.2.pwconv2.conv.weight', 'PPN.stages.0.1.conv.weight', 'PPN.stages.1.1.conv.weight', 'PPN.stages.2.1.conv.weight', 'PPN.stages.3.1.conv.weight', 'PPN.bottleneck.0.conv.weight', 'FPN.conv1x1.0.conv.weight', 'FPN.conv1x1.1.conv.weight', 'FPN.conv1x1.2.conv.weight', 'FPN.smooth_conv.0.conv.weight', 'FPN.smooth_conv.1.conv.weight', 'FPN.smooth_conv.2.conv.weight', 'FPN.conv_fusion.0.conv.weight',
        'resize_layers.0.conv.conv_transpose.weight', 'resize_layers.1.conv.conv_transpose.weight', 'projects.0.conv.weight', 'projects.1.conv.weight', 'projects.2.conv.weight', 'projects.3.conv.weight', 'resize_layers.3.conv.conv.weight', 'scratch.layer1_rn.conv.weight', 'scratch.layer2_rn.conv.weight', 'scratch.layer3_rn.conv.weight', 'scratch.layer4_rn.conv.weight', 'scratch.refinenet1.resConfUnit1.conv1.conv.weight', 'scratch.refinenet1.resConfUnit1.conv2.conv.weight', 'scratch.refinenet1.resConfUnit2.conv1.conv.weight', 'scratch.refinenet1.resConfUnit2.conv2.conv.weight', 'scratch.refinenet2.resConfUnit1.conv1.conv.weight', 'scratch.refinenet2.resConfUnit1.conv2.conv.weight', 'scratch.refinenet2.resConfUnit2.conv1.conv.weight', 'scratch.refinenet2.resConfUnit2.conv2.conv.weight', 'scratch.refinenet3.resConfUnit1.conv1.conv.weight', 'scratch.refinenet3.resConfUnit1.conv2.conv.weight', 'scratch.refinenet3.resConfUnit2.conv1.conv.weight', 'scratch.refinenet3.resConfUnit2.conv2.conv.weight', 'scratch.refinenet4.resConfUnit1.conv1.conv.weight', 'scratch.refinenet4.resConfUnit1.conv2.conv.weight', 'scratch.refinenet4.resConfUnit2.conv1.conv.weight', 'scratch.refinenet4.resConfUnit2.conv2.conv.weight',
        'out_conv1.conv.weight', 'out_conv2.0.conv.weight', 'out_conv2.2.weight'
    ]
    # bi_keys = [
    #     'backbone.downsample_layers.1.0.weight', 'backbone.downsample_layers.2.0.weight', 'backbone.downsample_layers.3.0.weight', 'backbone.stages.0.0.dwconv.weight', 'backbone.stages.0.0.pwconv1.weight', 'backbone.stages.0.0.pwconv2.weight', 'backbone.stages.0.1.dwconv.weight', 'backbone.stages.0.1.pwconv1.weight', 'backbone.stages.0.1.pwconv2.weight', 'backbone.stages.0.2.dwconv.weight', 'backbone.stages.0.2.pwconv1.weight', 'backbone.stages.0.2.pwconv2.weight', 'backbone.stages.1.0.dwconv.weight', 'backbone.stages.1.0.pwconv1.weight', 'backbone.stages.1.0.pwconv2.weight', 'backbone.stages.1.1.dwconv.weight', 'backbone.stages.1.1.pwconv1.weight', 'backbone.stages.1.1.pwconv2.weight', 'backbone.stages.1.2.dwconv.weight', 'backbone.stages.1.2.pwconv1.weight', 'backbone.stages.1.2.pwconv2.weight', 'backbone.stages.2.0.dwconv.weight', 'backbone.stages.2.0.pwconv1.weight', 'backbone.stages.2.0.pwconv2.weight', 'backbone.stages.2.1.dwconv.weight', 'backbone.stages.2.1.pwconv1.weight', 'backbone.stages.2.1.pwconv2.weight', 'backbone.stages.2.2.dwconv.weight', 'backbone.stages.2.2.pwconv1.weight', 'backbone.stages.2.2.pwconv2.weight', 'backbone.stages.2.3.dwconv.weight', 'backbone.stages.2.3.pwconv1.weight', 'backbone.stages.2.3.pwconv2.weight', 'backbone.stages.2.4.dwconv.weight', 'backbone.stages.2.4.pwconv1.weight', 'backbone.stages.2.4.pwconv2.weight', 'backbone.stages.2.5.dwconv.weight', 'backbone.stages.2.5.pwconv1.weight', 'backbone.stages.2.5.pwconv2.weight', 'backbone.stages.2.6.dwconv.weight', 'backbone.stages.2.6.pwconv1.weight', 'backbone.stages.2.6.pwconv2.weight', 'backbone.stages.2.7.dwconv.weight', 'backbone.stages.2.7.pwconv1.weight', 'backbone.stages.2.7.pwconv2.weight', 'backbone.stages.2.8.dwconv.weight', 'backbone.stages.2.8.pwconv1.weight', 'backbone.stages.2.8.pwconv2.weight', 'backbone.stages.3.0.dwconv.weight', 'backbone.stages.3.0.pwconv1.weight', 'backbone.stages.3.0.pwconv2.weight', 'backbone.stages.3.1.dwconv.weight', 'backbone.stages.3.1.pwconv1.weight', 'backbone.stages.3.1.pwconv2.weight', 'backbone.stages.3.2.dwconv.weight', 'backbone.stages.3.2.pwconv1.weight', 'backbone.stages.3.2.pwconv2.weight', 'PPN.stages.0.1.weight', 'PPN.stages.1.1.weight', 'PPN.stages.2.1.weight', 'PPN.stages.3.1.weight', 'PPN.bottleneck.0.weight', 'FPN.conv1x1.0.0.weight', 'FPN.conv1x1.1.0.weight', 'FPN.conv1x1.2.0.weight', 'FPN.smooth_conv.0.0.weight', 'FPN.smooth_conv.1.0.weight', 'FPN.smooth_conv.2.0.weight', 'FPN.conv_fusion.0.weight',
    #     'resize_layers.0.weight', 'resize_layers.1.weight', 'projects.0.weight', 'projects.1.weight', 'projects.2.weight', 'projects.3.weight', 'resize_layers.3.weight', 'scratch.layer1_rn.weight', 'scratch.layer2_rn.weight', 'scratch.layer3_rn.weight', 'scratch.layer4_rn.weight', 'scratch.refinenet1.resConfUnit1.conv1.weight', 'scratch.refinenet1.resConfUnit1.conv2.weight', 'scratch.refinenet1.resConfUnit2.conv1.weight', 'scratch.refinenet1.resConfUnit2.conv2.weight', 'scratch.refinenet2.resConfUnit1.conv1.weight', 'scratch.refinenet2.resConfUnit1.conv2.weight', 'scratch.refinenet2.resConfUnit2.conv1.weight', 'scratch.refinenet2.resConfUnit2.conv2.weight', 'scratch.refinenet3.resConfUnit1.conv1.weight', 'scratch.refinenet3.resConfUnit1.conv2.weight', 'scratch.refinenet3.resConfUnit2.conv1.weight', 'scratch.refinenet3.resConfUnit2.conv2.weight', 'scratch.refinenet4.resConfUnit1.conv1.weight', 'scratch.refinenet4.resConfUnit1.conv2.weight', 'scratch.refinenet4.resConfUnit2.conv1.weight', 'scratch.refinenet4.resConfUnit2.conv2.weight',
    #     'out_conv1.weight', 'out_conv2.0.weight', 'out_conv2.3.weight'
    # ]
    # # states = [state_dict[k].numel() for k in keys if not k.endswith('.conv.weight')]
    # # bi_states = [state_dict[k].numel() for k in keys if k.endswith('.conv.weight')]

    import pdb
    pdb.set_trace()
    states = [(k, state_dict[k].numel()) for k in keys if not (k in bi_keys or ('biconv' in k and 'weight' in k) 
                 or (k.endswith('fc1.weight') or k.endswith('fc2.weight') or k.endswith('qkv.weight') or k.endswith('proj.weight'))
                 or (k.startswith('backbone.norm') or k.startswith('backbone.head'))
                 )]
    bi_states = [(k, state_dict[k].numel()) for k in keys if k in bi_keys or ('biconv' in k and 'weight' in k) 
                 or (k.endswith('fc1.weight') or k.endswith('fc2.weight') or k.endswith('qkv.weight') or k.endswith('proj.weight'))
                 ]

    fp32_params = [x[1] for x in states]
    bi_params = [x[1] for x in bi_states]
    params = sum(fp32_params) + sum(bi_params) / 32
    pdb.set_trace()

    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    latest_checkpoint_callback = ModelCheckpoint()
    best_checkpoint_callback = ModelCheckpoint(monitor='valid/mIoU', mode='max')
    
    devices = [int(x) for x in args.gpus.split(',')]
    accumulate_grad_batches = max(1, config.TRAINING.BATCH_SIZE // config.TRAINING.BATCH_SIZE_ON_1_GPU // len(devices))

    trainer = L.Trainer(
        accelerator='gpu',
        # strategy='ddp_find_unused_parameters_true',
        # devices=devices,
        devices=[0],
        fast_dev_run=1,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=config.TRAINING.NUM_EPOCHS, 
        callbacks=[lr_monitor, latest_checkpoint_callback, best_checkpoint_callback],
        precision='32',
        check_val_every_n_epoch=config.TRAINING.EVAL_FREQ,
    )
    
    trainer.fit(segmentation_trainer, train_dataloader, valid_dataloader, ckpt_path=args.resume)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
