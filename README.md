# Vision Transformer for Domain Adaptation and Transfer Learning.

**For the original paper and source code please see:** [SHOT](https://github.com/tim-learn/SHOT)
**This Repo is a modified version:**

This repository contains the research code used to test the SHOT domain
adaptation technique with using different backbone networks, Like Vision Transformers, Swin Transformer, ConvNeXt, ResNet, HRnet.

## Setup

### Clone Repository

1. Clone repository
   ```
     git clone git@github.com:Meeeheeer/SHOT.git
   ```

### Requirements
See Dockerfile for full list of install dependencies and packages

- CUDA == 11.3.1
- torch == 1.10.2+cu113
- torchvision=0.11.3+cu113
- [apex](https://github.com/NVIDIA/apex)

#### Build Docker Container (Optional)

You may use the provided Dockerfile to build a container with all
of the necessary requirements required to run the provided code.
However, you must have some version of CUDA, Docker and the
NVIDIA container toolkit installed (see [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

1. Update .dockerignore with any added directories as necessary
2. Build the Docker container
    ```
   $ docker build -t <desired-tag> -f Dockerfile .
    ```
3. Run Docker Container.  `<tag>` must be the same tag used in step 4.
   ```
   docker run -it --gpus all --shm-size=25G -e HOME=$HOME -e USER=$USER -v $HOME:$HOME -w $HOME --user developer <tag>
   ```
4. Navigate to code (Home directories will be linked) and run

### Download Datasets
Datasets are loaded using the [DatasetFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder)
class.  Therefore all datasets should be downloaded in directories
as such:
```
directory/
├── File Name
    ├── class_x
    │   ├── xxx.ext
    │   ├── xxy.ext
    │   └── ...
    │       └── xxz.ext
    └── class_y
        ├── 123.ext
        ├── nsdf3.ext
        └── ...
        └── asd932_.ext
```

[utils/data.py](utils/data.py) is use to build the dataset and the dataloader. Please add dataset name as key and dataset folder name as value in `dataset_paths` present in [utils/data.py](utils/data.py). the dataset key can be passed in the any of the following files [image_source.py](object/image_source.py), [image_target.py](object/image_target.py), [image_eval.py](object/image_eval.py)   

Root file for the dataset can be passed by either setting `config.DATA.DATA_PATH` and `config.TARGET_DATA_PATH` or by passing `--data_path` and `--t-data-path` argument in the scripts.

The file `utils/partition_dota_xview.py` provides a helpful script
for partitioning a dataset into training/validation splits.

### CONFIGURATION FILE
[utils/config.py] is used to build base config file that is used to set [Dataset & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) classes, and train or test transforms in [utils/data.py](utils/data.py), and initialize optimizers parameters, data loading parameters in [image_source.py](object/image_source.py),[image_target.py](object/image_target.py). Certain Config parameters can be changed using arguments provided in the scripts, [utils/config.py](utils/config.py) contains more information on how to set the configuration parameters.

**NOTE: Change the base configuration in [utils/config.py](utils/config.py) before use, the config file given to any script will modify the base config set.**




### Backbone Models and Imagnet Pretrained Weights
To recreate the results pass the following key in --net argument in  any script. All backnone models and pretrained weights are taken from [timm](https://github.com/rwightman/pytorch-image-models), all models and pretrained weights avaiable in [timm](https://github.com/rwightman/pytorch-image-models) is listed under the following [documentation](https://rwightman.github.io/pytorch-image-models/models/) or can viewed using [timm.list_models](https://rwightman.github.io/pytorch-image-models/#list-models-with-pretrained-weights). `--net` argument is used to intialized the backbone model in all scripts [image_source.py](object/image_source.py),[image_target.py](object/image_target.py),[image_eval.py](object/image_eval.py). `--pretrained` argument can be used to manually load netF weights in any of the scripts.

The following models and pretrained weights used for this project are given below

* [resnet50]()
* [hrnet_w48]()
* [convnext_base_384_in22ft1k](https://github.com/rwightman/pytorch-image-models/blob/0.6.x/timm/models/convnext.py)
* [vit_base_patch16_384](https://github.com/rwightman/pytorch-image-models/blob/0.6.x/timm/models/vision_transformer.py)
* [swin_base_patch4_window12_384](https://github.com/rwightman/pytorch-image-models/blob/0.6.x/timm/models/swin_transformer.py)

*Any script that loads netF, netB, and netC only needs to be
pointed to the saved netF path using the `--pretrained` argument*

## Run the Code
Sample config files and scripts are contained in 
[sample_configs](sample_configs) and [sample_scripts](sample_scripts)
respectively.

### Fine Tune Model on Target Dataset
The file [image_source.py](object/image_source.py) is used
to fine tune a model onto a source dataset.  To fine tune a 
Swin-B model pretrained on ImageNet-22k to DOTA you could run
the following command:

[--mode] is used to split the dataset into training and validation dataset for fine-tuning on the validation dataset. Set to 'all' to use the full dataset for training, when set to 'all' the target dataset is used to fine-tune in [image_source.py](object/image_source.py). Set to 'split' to split the source dataset in [image_source.py] into training and validation split. 

If training and validation data are in seprate folders, dataset name key for both folders can be set in [utils/data.py], which will allow the script to load training data folder as source and validation data folder as target dataset.

*NOTE:
[image_source.py] uses `--dset` and `--datat_path` for source, with `--t-dset`, `--t-data-path` for target, these arguments overides `config.DATA.DATASET` and `config.DATA.TARGET_DATASET`  feilds in config file passed. `--mode='all'` use both datasets source and target, `--mode='split'` uses only source specified in `--dset` and `--data_path`

[image_target.py] uses only `--t-dset` and `--t-data-path` for adaptation, to point to target dataset, it uses the whole target set with true labels for validation on adaptation.`--mode=split` can be given to split the dataset on target file.
*
 

```
PYTHONPATH=.:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_source.py \
--trte val \
--da uda \
--output output \
--gpu_id 0 \
--cfg sample_configs/example_source.yaml \
--dset dota \
--data-path /data \
--t-dset xview \
--t-data-path ~/data \
--evals-per-epoch 1 \
--batch_size=20 \
--net=swin_base_patch4_window12_384 \
--mode 'all' \
--transfer-dataset \
--source 1 \
--target 0 \
--name=dota-source-1
```

Exampled of using `--mode='split` and `pretrained`

```
PYTHONPATH=.:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_source.py \
--trte val \
--da uda \
--output output \
--gpu_id 0 \
--cfg sample_configs/example_source.yaml \
--pretrained ~/SHOT/output/path_to_weights \
--dset dota \
--data-path ~/data \
--evals-per-epoch 1 \
--batch_size=20 \
--net=swin_base_patch4_window12_384 \
--mode 'split' \
--transfer-dataset \
--source 1 \
--target 0 \
--name=swin-dota-source-1
```

For this code, the `TOP_N` performing models on the target
domain will be saved in `output/<name>/<net>/` for further analysis.
The `TOP_N` value is a global variable in the `image_source.py`
script.

### Evaluate Model Generalization
The file [image_eval.py](object/image_eval.py) is used to 
evaluate the performance of a model on both a source and target
dataset.  This script does not perform any training and is for
evaluation only.

To evaluate the performance of the Swin-B model fine-tuned on
the DOTA dataset you could run the following command:

```
PYTHONPATH=.:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_eval.py \
--output output \
--gpu_id 0 \
--cfg sample_configs/config_file_use \
--pretrained ~/SHOT/output/path_to_weights \
--dset dota \
--data-path ~/data/dota-xview/ \
--t-dset xview \
--t-data-path ~/data/dota-xview/ \
--batch_size=128 \
--net=swin_base_patch4_window12_384 \
--transfer-dataset \
--source 1 \
--target 0 \
--name=swin-dota-source-1
```



Note that the saved output of this evaluation is placed in 
`output/eval/<name>`.

Also if the `--dset` and `--data-path` arguments are omitted
this script can simply evaluate the model on a given target dataset.

`-t-data-path` is given in all scripts if target folder root is different from source folder root. simply passing `--data-path` will work for all scripts if target dataset is in the same folder as source dataset.


**NOTE: When given source dataset the given script will also save t-SNE plot for source and target feature analysis, otherwise it will only save the classfification report with class wise t-SNE plot on target.**

### Adapt model using SHOT to Target Domain
The script [image_target.py](object/image_target.py) is used to 
adapt a given model onto a target domain using the unsupervised 
domain adaptation technique SHOT.

To perform this adaptation on a Swin-B model fine-tuned on DOTA
and adapt it to the XVIEW dataset, the following command
could be used:

```
PYTHONPATH=.:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_target.py \
--cls_par 0.3 \
--da uda \
--output output \
--gpu_id 0 \
--cfg sample_configs/example_target \
--pretrained output/swin-dota-source-1/V/ckpt_epoch_9_eval_8.pth \
--t-dset xview \
--t-data-path ~/data/dota-xview/ \
--batch_size=20 \
--evals-per-epoch=2 \
--net=swin_base_patch4_window12_384 \
--transfer-dataset \
--source -1 \
--target 0 \
--name=dota-to-xview-1
```


### Tensorboard
Important training metrics for this project are logged using
[Tensorboard](https://www.tensorflow.org/tensorboard/get_started).
When training these metrics can be seen by:

1. `$ tensorboard --logdir='logs/<name>`
2. Open a web-browser and navigate to `localhost:6006`

