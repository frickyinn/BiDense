import argparse
import lightning as L

from configs.depth.default import get_cfg_defaults
from datasets.depth_dataloader import get_dataLoader
from pl_trainer_depth import PL_DepthTrainer


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

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
    test_dataloader = get_dataLoader(mode='online_eval', **dataloader_args)
    depth_trainer = PL_DepthTrainer.load_from_checkpoint(args.ckpt_path)
    
    devices = [int(x) for x in args.gpus.split(',')]
    trainer = L.Trainer(
        accelerator='gpu',
        devices=devices,
        precision='32',
    )
    
    trainer.test(depth_trainer, dataloaders=test_dataloader)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--gpus', type=str, default='0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
