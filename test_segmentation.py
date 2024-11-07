import argparse
import lightning as L

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
    test_dataloader = get_dataLoader(mode='val', **dataloader_args)
    segmentation_trainer = PL_SegmentationTrainer.load_from_checkpoint(args.ckpt_path)
    
    devices = [int(x) for x in args.gpus.split(',')]
    trainer = L.Trainer(
        accelerator='gpu',
        devices=devices,
        precision='32',
    )
    
    trainer.test(segmentation_trainer, test_dataloader)


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
