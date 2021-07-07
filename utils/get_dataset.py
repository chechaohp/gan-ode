from dataset import UCF101


def get_training_set(cfg, spatial_transform, temporal_transform,
                     target_transform):
    assert cfg.DATASET.NAME in ['ucf101']

    if cfg.DATASET.NAME == 'ucf101':
        training_data = UCF101(
            cfg.DATASET.VIDEO_PATH,
            cfg.DATASET.ANNOTATION_PATH,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    
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