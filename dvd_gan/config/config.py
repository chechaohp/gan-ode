from yacs.config import CfgNode as CN

_C = CN()


_C.DATASET = CN()
_C.DATASET.NAME = 'ucf101'
_C.DATASET.N_CLASS = 101
_C.DATASET.K_SAMPLE = 64
_C.DATASET.N_FRAMES = 24
_C.DATASET.ROOT_PATH = './UCF101'
_C.DATASET.IMAGE_PATH = './data'
_C.DATASET.VIDEO_PATH = './video_jpeg'
_C.DATASET.ANNOTATION_PATH = './annotation/ucf101_01.json'


_C.MODEL = CN()
_C.MODEL.NAME = 'dvd_gan'
_C.MODEL.IMAGE_SIZE = 128
_C.MODEL.G_NUM = 5
_C.MODEL.Z_DIM = 120
_C.MODEL.G_CHANNEL = 32
_C.MODEL.DS_CHANNEL = 32
_C.MODEL.DT_CHANNEL = 32
_C.MODEL.N_FRAMES = 8
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.D_CONV_DIM = 64

# Log
_C.LOG = CN()
_C.LOG.LOG_PATH = './logs'
_C.LOG.LOG_EPOCH = 1
_C.LOG.MODEL_SAVE_PATH = './save_models'
_C.LOG.MODEL_SAVE_EPOCH = 200
_C.LOG.SAMPLE_PATH = './samples'
_C.LOG.SAMPLE_EPOCH = 20


_C.TRAINING = CN()
_C.TRAINING.LOSS = 'wgan-gp'
_C.TRAINING.LAMBDA_GP = 10 # gradient penalty coef
_C.TRAINING.TOTAL_EPOCH = 100000
_C.TRAINING.D_ITERS = 1
_C.TRAINING.G_ITERS = 1
_C.TRAINING.BATCH_SIZE = 8
_C.TRAINING.NUM_WORKSERS = 8
_C.TRAINING.G_LR = 5e-5
_C.TRAINING.D_LR = 5e-5
_C.TRAINING.LR_DECAY = 0.9999
_C.TRAINING.BETA1 = 0.0
_C.TRAINING.BETA2 = 0.9
_C.TRAINING.PRETRAIN = None
_C.TRAINING.TRAIN_CROP = 'corner'
_C.TRAINING.LR_SCHR = 'const'

