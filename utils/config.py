"""
Modified from: https://github.com/ddp5730/Swin-Transformer/blob/d3f1cf396db65d666525bd522d6d19c283cf9529/data/build.py

"""



import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
_C.DATA.TARGET_DATASET = None
# Dataset name
_C.DATA.DATASET = ''
_C.DATA.TARGET_DATASET = ''
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4 

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.NAME = ''
# Enter all Model architecture configs here
_C.MODEL.CONFIG = CN()
# Drop path rate
_C.MODEL.CONFIG.DROP_PATH_RATE = 0.0
# Intialized to one class, change using config file.
_C.MODEL.NUM_CLASSES = 1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Randomly flip images horzontally given an probability
_C.AUG.RANDOM_HORIZONTAL_FLIP = 0.
# Randomly flip images vertically given an probability
_C.AUG.RADOM_VERTICAL_FLIP = 0.
# Mixup alpha, mixup enabled if > 0
#change to alpha
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
#change to cutmix aplha
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to transfer this model from one dataset to another
_C.TRANSFER_DATASET = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0



def update_model_config(net,config):
    model_configs = {'vit-b':{'train':{'EPOCHS':20,
                        'WARMUP_EPOCHS':2,
                        'BASE_LR':1e-04},
                        'model':{'config':{'DROP_PATH_RATE':0.1},
                                    'pretrained':'/home/mm7144/DA_ViT/SHOT/output/OfficeHome/pr/genralization/lr_1e4_wd_2e6_batch_32_cvnxt_swin_vit_only/vit-b/source_F_epoch_18_eval_17.pt'}},
                    'hrnet':{'data': {'img_size': 224 }
                        ,'train':{'EPOCHS':30,
                        'WARMUP_EPOCHS':20,
                        'BASE_LR':2e-04},
                        'model':{'pretrained':'/home/mm7144/DA_ViT/SHOT/output/OfficeHome/cl/genralization/use_this_hrnet/hrnet/source_F_epoch_6_eval_5.pt'}},
                    'swin-b':{'train':{'EPOCHS':20,
                        'WARMUP_EPOCHS':2,
                        'BASE_LR':1e-04},
                        'model':{'config':{'DROP_PATH_RATE':0.1},
                                    'pretrained':'/home/mm7144/DA_ViT/SHOT/output/cub199/genralization/swin-b/source_F_epoch_10_eval_9.pt'}},
                    'cvnxt-b':{'train':{'EPOCHS':20,
                        'WARMUP_EPOCHS':2,
                        'BASE_LR':1e-04},
                        'model':{'pretrained':'/home/mm7144/DA_ViT/SHOT/output/OfficeHome/ar/genralization/cvnxt_vit_swin_final_use_this/cvnxt-b/source_F_epoch_7_eval_6.pt'}},
                    'res-50':{'data': {'img_size': 224 },
                    'train':{'EPOCHS':30,
                        'WARMUP_EPOCHS':15,
                        'BASE_LR':1e-04},
                        'model':{'pretrained':'/home/mm7144/DA_ViT/SHOT/output/cub199/genralization/res-50/source_F_epoch_28_eval_27.pt'}}}
    
    if net in model_configs:
        config['MODEL']['NAME'] = net
        for k,v in model_configs[net]['train'].items():
            config['TRAIN'][k] = v
        if 'model' in model_configs[net]:
            if 'config' in model_configs[net]['model']:
                for k,v in model_configs[net]['model']['config'].items():
                    config['MODEL']['CONFIG'][k] = v
            if 'pretrained' in model_configs[net]['model']:
                config['MODEL']['PRETRAINED'] = model_configs[net]['model']['pretrained']
        if 'data' in model_configs[net]:
            if 'img_size' in model_configs[net]['data']:
                config['DATA']['IMG_SIZE'] = model_configs[net]['data']['img_size']        
    
def _update_config_from_file(config, cfg_file):
    '''
    Update Models from config.
    Args:
        config: base configuration
        cfg_file: configuration file to add updates.
    
    return: None
    '''
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    '''
    Update configs with cmd arguments.
    Args:
        config: base configuration.
        args: arguments to update base config
    
    return: None
    '''
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    config.MODEL.NAME =  args.net
    # merge from specific arguments
    if args.acc_iter:
        config.TRAIN.ACCUMULATION_STEPS = args.acc_iter
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.acc_iter:
        config.TRAIN.ACCUMULATION_STEPS = args.acc_iter
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.transfer_dataset:
        config.TRAIN.TRANSFER_DATASET = True
    if args.eval_period:
        config.TRAIN.EVAL_PERIOD = args.eval_period
    
    if args.t_dset:
        config.DATA.TARGET_DATASET = args.t_dset


    if args.t_data_path:
        config.DATA.TARGET_DATA_PATH = args.t_data_path
    else:
        config.DATA.TARGET_DATA_PATH = config.DATA.DATA_PATH

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
