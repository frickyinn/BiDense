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
        batch_size=config.TRAINING.BATCH_SIZE_ON_1_GPU,
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

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    latest_checkpoint_callback = ModelCheckpoint()
    best_checkpoint_callback = ModelCheckpoint(monitor='valid/mIoU', mode='max')
    
    devices = [int(x) for x in args.gpus.split(',')]
    accumulate_grad_batches = max(1, config.TRAINING.BATCH_SIZE // config.TRAINING.BATCH_SIZE_ON_1_GPU // len(devices))

    trainer = L.Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=devices,
        # devices=[0],
        # fast_dev_run=1,
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
