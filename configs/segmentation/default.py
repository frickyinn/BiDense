from yacs.config import CfgNode as CN


_CN = CN()

# Dataset
_CN.DATASET = CN()
_CN.DATASET.DATASET = None
_CN.DATASET.DATA_PATH = None
_CN.DATASET.BASE_SIZE = 520
_CN.DATASET.CROP_SIZE = 480

# Training
_CN.TRAINING = CN()
_CN.TRAINING.NUM_THREADS = 1
_CN.TRAINING.BATCH_SIZE = 32
_CN.TRAINING.BATCH_SIZE_ON_1_GPU = 8
_CN.TRAINING.NUM_EPOCHS = 50
_CN.TRAINING.MAX_LR = None

# Model
_CN.MODEL = CN()
_CN.MODEL.MODEL_TYPE = None
_CN.MODEL.BINARY_TYPE = None
_CN.MODEL.NUM_CLASSES = None

_CN.RANDOM_SEED = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
