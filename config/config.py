from yacs.config import CfgNode as CN

_C = CN()
_C.VERSION = '1'

_C.DATASET = CN()
_C.DATASET.NAME = 'ucf101'
_C.DATASET.N_CLASS = 2
_C.DATASET.K_SAMPLE = 64
_C.DATASET.N_FRAMES = 24
_C.DATASET.ROOT_PATH = '../UCF101'
_C.DATASET.IMAGE_PATH = 'data'
_C.DATASET.VIDEO_PATH = 'videos_jpeg'
_C.DATASET.ANNOTATION_PATH = 'annotations/ucf101_01.json'
_C.DATASET.NORM_VALUE = 255
_C.DATASET.NO_MEAN_NORM = True
_C.DATASET.STD_NORM = False
_C.DATASET.MEAN_DATASET = 'activitynet'
_C.DATASET.INITIAL_SCALE = 1
_C.DATASET.N_SCALES = 5
_C.DATASET.SCALE_STEP = 0.84089641525
_C.DATASET.SAMPLE_SIZE = 64


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
_C.LOG.SAMPLE_EPOCH = 200


_C.TRAIN = CN()
_C.TRAIN.MODE = True
_C.TRAIN.LOSS = 'hinge'
_C.TRAIN.LAMBDA_GP = 10 # gradient penalty coef
_C.TRAIN.TOTAL_EPOCH = 100000
_C.TRAIN.D_ITERS = 1
_C.TRAIN.G_ITERS = 1
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.G_LR = 5e-5
_C.TRAIN.D_LR = 5e-5
_C.TRAIN.LR_DECAY = 0.9999
_C.TRAIN.BETA1 = 0.0
_C.TRAIN.BETA2 = 0.9
_C.TRAIN.PRETRAIN = None
_C.TRAIN.TRAIN_CROP = 'corner'
_C.TRAIN.LR_SCHR = 'const'
_C.TRAIN.GPUS = ['0']
_C.TRAIN.PARALLEL = False

_C.TEST = CN()
_C.TEST.TEST_SUBSET = 'val'
_C.TEST.N_VAL_SAMPLE = 1

_C.OTHER = CN()
_C.OTHER.USE_TENSORBOARD = False
_C.OTHER.TEST_BATCH_SIZE = 4