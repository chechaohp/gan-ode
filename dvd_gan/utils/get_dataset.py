from dataset import UCF101


def get_training_set(cfg, spatial_transform, temporal_transform,
                     target_transform):
    assert cfg.dataset in ['ucf101']

    if cfg.dataset == 'ucf101':
        training_data = UCF101(
            cfg.video_path,
            cfg.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    
    return training_data


def get_validation_set(cfg, spatial_transform, temporal_transform,
                       target_transform):
    assert cfg.dataset in ['ucf101']

    if cfg.dataset == 'ucf101':
        validation_data = UCF101(
            cfg.video_path,
            cfg.annotation_path,
            'validation',
            cfg.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=cfg.n_frames)
    return validation_data


def get_test_set(cfg, spatial_transform, temporal_transform, target_transform):
    assert cfg.dataset in ['ucf101']
    assert cfg.test_subset in ['val', 'test']

    if cfg.test_subset == 'val':
        subset = 'validation'
    elif cfg.test_subset == 'test':
        subset = 'testing'
    
    if cfg.dataset == 'ucf101':
        test_data = UCF101(
            cfg.video_path,
            cfg.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=cfg.n_frames)
    return test_data