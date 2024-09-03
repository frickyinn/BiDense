import argparse
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from configs.depth.default import get_cfg_defaults
from misc import seed_torch
from datasets.depth_dataloader import get_dataLoader
from pl_trainer import PL_DepthTrainer


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    seed = config.RANDOM_SEED
    seed_torch(seed)

    dataloader_args = dict(
        batch_size=config.TRAINING.BATCH_SIZE_ON_1_GPU, 
        num_threads=config.TRAINING.NUM_THREADS,

        dataset=config.DATASET.DATASET,
        data_path=config.DATASET.DATA_PATH,
        gt_path=config.DATASET.GT_PATH,
        filenames_file=config.DATASET.FILENAMES_FILE,
        data_path_eval=config.DATASET.DATA_PATH_EVAL,
        gt_path_eval=config.DATASET.GT_PATH_EVAL,
        filenames_file_eval=config.DATASET.FILENAMES_FILE_EVAL,
        input_height=config.DATASET.INPUT_HEIGHT,
        input_width=config.DATASET.INPUT_WIDTH,

        do_random_rotate=config.PREPROCESSING.DO_RANDOM_ROTATE,
        degree=config.PREPROCESSING.DEGREE,
        do_kb_crop=config.PREPROCESSING.DO_KB_CROP,
        use_right=config.PREPROCESSING.USE_RIGHT,
    )
    train_dataloader = get_dataLoader(mode='train', **dataloader_args)
    valid_dataloader = get_dataLoader(mode='online_eval', **dataloader_args)
    
    trainer_args = dict(
        model_type=config.MODEL.MODEL_TYPE,
        binary_type=config.MODEL.BINARY_TYPE,
        dataset=config.DATASET.DATASET,
        variance_focus=config.TRAINING.VARIANCE_FOCUS,
        max_lr=config.TRAINING.MAX_LR,
        epochs=config.TRAINING.NUM_EPOCHS,
        garg_crop=config.ONLINE_EVAL.GARG_CROP,
        eigen_crop=config.ONLINE_EVAL.EIGEN_CROP,
        min_depth_eval=config.ONLINE_EVAL.MIN_DEPTH_EVAL,
        max_depth_eval=config.ONLINE_EVAL.MAX_DEPTH_EVAL,
        max_depth=config.MODEL.MAX_DEPTH
    )
    if args.weights is None:
        depth_trainer = PL_DepthTrainer(**trainer_args)
    else:
        depth_trainer = PL_DepthTrainer.load_from_checkpoint(checkpoint_path=args.weights, **trainer_args)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    latest_checkpoint_callback = ModelCheckpoint()
    best_checkpoint_callback = ModelCheckpoint(monitor='valid/d1', mode='max')
    
    devices = [int(x) for x in args.gpus]
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
        precision='32' if config.MODEL.BINARY_TYPE == 'fp32' else 'bf16-mixed',
        check_val_every_n_epoch=config.ONLINE_EVAL.EVAL_FREQ,
    )
    
    trainer.fit(depth_trainer, train_dataloader, valid_dataloader, ckpt_path=args.resume)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--gpus', type=list, default=[0, 1])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
