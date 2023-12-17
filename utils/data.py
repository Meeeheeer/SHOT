import os
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from sklearn.model_selection import train_test_split



try:
    from torchvision.transforms import InterpolationMode


    def str_to_pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import str_to_pil_interp







dataset_paths = {'aid':'Source_AID',
                'ucm':'Target_UCM',
                'clrs':'CLRS',
                'npwu':'NPWU',
                'dota':'DOTA',
                'xview':'XVIEW',
                'rareplane-r':'',
                'rareplane-s':'',
                'cub-200':'CUB_200_2011/images',
                'cub-pa':'CUB-200-Painting'}



multi_domain_datasets = {'OfficeHome':{'name':'OfficeHomeDataset_10072016',
                                        'cl':'Clipart',
                                        'rw':'Real World',
                                        'ar':'Art',
                                        'pr':'Product'},
                        'domainet':{'name':'Domainet',
                                        'cl':'Clipart',
                                        'rw':'Real-World',
                                        'ar':'Art',
                                        'pr':'Peoduct',
                                        'qd':'QuickDraw',
                                        'ig':'Infograph'}}

def build_loader(config,mode,dset,target=False,root=None):
    """
    Builds dataloader and mixup function.
    Args:
        config : yaml config to set augmentations, dataloader params
        mode : 'all' to use full dataset, 'plit' to retrun a split
        dset: key for dataset_path and multi_domain_datasets.
        target: 'False' will return train transforms, True will return test transforms
        root: root file where datasets are stored.
    
    Returns: Dataloader or Train Dataloader, Validation Dataloader.
    """
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    if mode == 'split':
        #loading dataset, target is set to chnage the transforms to non augmented transforms
        dataset_train, dataset_val = build_dataset(mode = mode,config = config,tr = not target, dset = dset, root_path=None)
        
        #samplers
        sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle= not target,drop_last= not target)
        sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle= False, drop_last= False)

        #loaders
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successally build dataset")
        data_loader_train = DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last= not target,
        )
        data_loader_val = DataLoader(
                                    dataset_val, sampler=sampler_val,
                                    batch_size=config.DATA.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=config.DATA.PIN_MEMORY,
                                    drop_last=False
                                )
        
        return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

    else:
        dataset_train = build_dataset(mode=mode,dset=dset,tr= not target, config=config,root_path=root)
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle= not target,drop_last= not target
        )
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successally build dataset")
        data_loader_train = DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last= not target,
        )
        return dataset_train, data_loader_train, mixup_fn

    


def build_dataset(mode,config,tr,dset,root_path=None,split_size=0.8):
    """
    Reads the dataset in ImageFolder format. root is set in 
    config under 'config.DATA.DATA_PATH' or custom argument root_path 
    can be used to set root to different file directory to read data.
    
    mode = 'all' will return the full dataset
    mode = 'split' will return train and val datset split according to split_size.

    All datasets must be instializes in dataset_path dict, use the following key value pair format given below 
    'DATASET 1 KEY NAME':'File name of Dataset 1',
    'DATASET 2 KEY':'File name of Dataset 2':,...

    For using the multidomain dataset, use the following fashion key vakue pair format given below.
    Note Dataset key name and domian key name must seprated with '-' in the following manner DATASET_KEY_NAME-DOMAIN_KEY_NAME
    'DATASET KEY NAME': {'name':'File name under which all domains reside',
                        'Domain 1 key':'File name of Domain 1',
                        Domain 2 key':'File name of Domain 2':,...}

    Args:
        mode : Is used to set the split flag for spliting the data.
        config : yaml config, is used to set the dataset root path.
        tr :  Used to set transform type. train or test
        dset : name of the dataset key to be used.
        root_path : Used to set custom root_path.
                    Note : This argument overides the data path set by config.DATA.DATA_PATH.
        split_size : Used to set the dataset split. By deafult set to 80%
    
    returns : dataset or train_dataset, validation dataset 
    """
    global dataset_paths
    global multi_domain_datasets

    root = root_path if root_path else config.DATA.DATA_PATH 

    transforms = build_transform(tr, config)

    if dset in dataset_paths.keys():
        root = os.path.join(root,dataset_paths[dset])
        dataset = datasets.ImageFolder(root,transform=transforms)


    if '-' in dset:
        n , k = dset.split('-')

        if n in multi_domain_datasets:
            name = multi_domain_datasets[n]['name']
            if k in multi_domain_datasets[n]:
                key = multi_domain_datasets[n][k]
                root = os.path.join(root,name,key)  
                dataset = datasets.ImageFolder(root,transform=transforms)
                    
    
    
    
    if mode == 'split':
        train_indices, val_indices = train_indices, valid_test_indices = train_test_split(np.arange(len(dataset)),
                                    train_size=split_size,
                                    stratify=dataset.targets,
                                    random_state=42)
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, val_indices)
        print('DATASET SIZE :',len(dataset))
        print('TRAINING DATASET SIZE :' ,len(train_dataset))
        print('VALIDATION DATASET SIZE :',len(valid_dataset))
        if tr:
            valid_dataset.dataset.transform = build_transform(not tr,config)
        return train_dataset, valid_dataset
    else:
        return dataset


def build_transform(tr, config):
    """
    Gets Training and testing trasnforms for dataset.
    Args:
        tr : Set true to get train transforms, set false to get test transforms
        config : yaml config file to set augmentations for transforms.

    returns : transforms 
    """
    resize_im = config.DATA.IMG_SIZE > 32
    if tr:
        # this should always dispatch to train transforms
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            hflip = config.AUG.RANDOM_HORIZONTAL_FLIP,
            vflip = config.AUG.RADOM_VERTICAL_FLIP
        )
       
        return transform
    
    t = [transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)]

    return transforms.Compose(t)



