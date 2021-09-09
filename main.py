from utils.trainer import Trainer
from utils.trainer_ode import Trainer as TrainerODE
# from tester import Tester
from dataset.data_loader import Data_Loader
from torch.backends import cudnn
from utils.utils import make_folder
from config import cfg

import os
import torch


##### Import libary for dataloader #####
##### https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/main.py
from dataset.transform.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from dataset.transform.temporal_transforms import LoopPadding, TemporalRandomCrop
from dataset.transform.target_transforms import ClassLabel, VideoID
from dataset.transform.target_transforms import Compose as TargetCompose
from utils.get_dataset import get_training_set, get_validation_set, get_test_set
from dataset.mean import get_mean

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='choose experiment setup.')
    parser.add_argument('-e', '--experiment_file', default='experiments/default.yml',help="Student training config")
    args = parser.parse_args()
    return args

def main():
    # For fast training
    cudnn.benchmark = True
    # update config file
    args = get_args()
    cfg.merge_from_file(args.experiment_file)
    ##### Dataloader #####
    if cfg.DATASET.NAME == 'ucf101':
        cfg.DATASET.VIDEO_PATH = os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.VIDEO_PATH)
        cfg.DATASET.ANNOTATION_PATH = os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.ANNOTATION_PATH)
    cfg.DATASET.MEAN = get_mean(cfg.DATASET.NORM_VALUE, dataset=cfg.DATASET.MEAN_DATASET)

    if cfg.DATASET.NO_MEAN_NORM and not cfg.DATASET.STD_NORM:
        norm_method = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif not cfg.DATASET.STD_NORM:
        norm_method = Normalize(cfg.mean, [1, 1, 1])

    cfg.DATASET.SCALES = [cfg.DATASET.INITIAL_SCALE]
    for i in range(1, cfg.DATASET.N_SCALES):
        cfg.DATASET.SCALES.append(cfg.DATASET.SCALES[-1] * cfg.DATASET.SCALE_STEP)

    if cfg.TRAIN.MODE:
        assert cfg.TRAIN.TRAIN_CROP in ['random', 'corner', 'center']
        if cfg.TRAIN.TRAIN_CROP == 'random':
            crop_method = MultiScaleRandomCrop(cfg.DATASET.SCALES, cfg.DATASET.SAMPLE_SIZE)
        elif cfg.TRAIN.TRAIN_CROP == 'corner':
            crop_method = MultiScaleCornerCrop(cfg.DATASET.SCALES, cfg.DATASET.SAMPLE_SIZE)
        elif cfg.TRAIN.TRAIN_CROP == 'center':
            crop_method = MultiScaleCornerCrop(
                cfg.DATASET.SCALES, cfg.DATASET.SAMPLE_SIZE, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(cfg.DATASET.NORM_VALUE), norm_method
        ])
        temporal_transform = TemporalRandomCrop(cfg.DATASET.N_FRAMES)
        target_transform = ClassLabel()

        print("="*30,"\nLoading data...")
        training_data = get_training_set(cfg, spatial_transform,
                                         temporal_transform, target_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True)
    else:
        spatial_transform = Compose([
            Scale(cfg.DATASET.SAMPLE_SIZE),
            CenterCrop(cfg.DATASET.SAMPLE_SIZE),
            ToTensor(cfg.DATASET.NORM_VALUE), norm_method
        ])
        temporal_transform = LoopPadding(cfg.DATASET.N_FRAMES)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            cfg, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKSERS,
            pin_memory=True)

    ##### End dataloader #####

    # Use Big-GAN dataset to test only
    # The random data is used in the trainer
    # Need to pre-process data and use the dataloader (above)

    # cfg.n_class = len(glob.glob(os.path.join(cfg.root_path, cfg.video_path)))

    ## Data loader
    print('number class:', cfg.DATASET.N_CLASS)
    # # Data loader
    # data_loader = Data_Loader(cfg.train, cfg.dataset, cfg.image_path, cfg.imsize,
    #                          cfg.batch_size, shuf=cfg.train)

    # Create directories if not exist
    make_folder(cfg.LOG.MODEL_SAVE_PATH, cfg.VERSION)
    make_folder(cfg.LOG.SAMPLE_PATH, cfg.VERSION)
    make_folder(cfg.LOG.LOG_PATH, cfg.VERSION)

    if cfg.TRAIN.MODE:
        if cfg.MODEL.NAME=='dvd_gan':
            trainer = Trainer(train_loader, cfg) 
        else:
            trainer = TrainerODE(train_loader,cfg)

        trainer.train()
    else:
        tester = Tester(val_loader, cfg)
        tester.test()


if __name__ == '__main__':
    # print(cfg)
    main()