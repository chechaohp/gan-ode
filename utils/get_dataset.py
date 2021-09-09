from dataset import UCF101, UCF101Video


def get_training_set(cfg, spatial_transform, temporal_transform,
                     target_transform):
    assert cfg.DATASET.NAME in ['ucf101','ucf101-video']

    if cfg.DATASET.NAME == 'ucf101':
        training_data = UCF101(
            cfg.DATASET.VIDEO_PATH,
            cfg.DATASET.ANNOTATION_PATH,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif cfg.DATASET.NAME == 'ucf101-video':
        training_data = UCF101Video(
            cfg.DATASET.ROOT_PATH,
            cfg.DATASET.VIDEO_PATH,
            cfg.DATASET.ANNOTATION_PATH,
            frames_per_clip=20,
            transform=spatial_transform
        )
    
    return training_data


def get_validation_set(cfg, spatial_transform, temporal_transform,
                       target_transform):
    assert cfg.DATASET.NAME in ['ucf101']

    if cfg.DATASET.NAME == 'ucf101':
        validation_data = UCF101(
            cfg.DATASET.VIDEO_PATH,
            cfg.DATASET.ANNOTATION_PATH,
            'validation',
            cfg.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=cfg.DATASET.N_FRAMES)
    return validation_data


def get_test_set(cfg, spatial_transform, temporal_transform, target_transform):
    assert cfg.DATASET.NAME in ['ucf101']
    assert cfg.TEST.TEST_SUBSET in ['val', 'test']

    if cfg.TEST.TEST_SUBSET == 'val':
        subset = 'validation'
    elif cfg.TEST.TEST_SUBSET == 'test':
        subset = 'testing'
    
    if cfg.DATASET.NAME == 'ucf101':
        test_data = UCF101(
            cfg.DATASET.VIDEO_PATH,
            cfg.DATASET.ANNOTATION_PATH,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=cfg.DATASET.N_FRAMES)
    return test_data