# image_source.py
# This file is used to train/finetune a backbone model on the target dataset in preparation
# for use with the SHOT domain adaptation method.
#
# This file will save the `TOP_N` models as evaluated by accuracy on the target dataset.
import argparse
import copy
import os

# Enter number of GPU's to be used before run, comment out if using a cluster.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path as osp
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm
import loss
import network
from network import get_backbone
from data_list import ImageList
from loss import CrossEntropyLabelSmooth
from utils.config import get_config
from utils.data import build_loader
from utils.logger import create_logger
from center_loss import CenterLoss

TOP_N = 1
ACC_ITER = 4

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)





def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """
    (Not used) Get Learning rate scheduler as used by the initial SHOT code.
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    """
    (Not used) Get Train transform as used by the initial SHOT code
    """
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    """
    (Not used) Get Test transform as used by the initial SHOT code
    """
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    """
    (Not Used) Load the data as used by the initial SHOT code.
    Args:
        args: Command line arguments

    Returns: Relevant dataloaders in a dictionary object

    """
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag = False):
    """
    Old method for calculating model performance by original SHOT code
    Args:
        loader: Test dataloader
        netF: Backbone network
        netB: Bottleneck network
        netC: Classification network
        flag: retuns F1-score and accuracy when set true.

    Returns: Model accuracy

    """
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_source(args, config):
    """
    Large method to set up model pretraining on secified source dataset and 
    fine-tune on specifed target dataset.

    The methond also uses split mode set by args.mode = 'all' or 'split',
    which will split the specifed source dataset under 'config.DATA_DATASET'
    and use path under 'config.DATA_PATH'.

    netF trained on source pretrained weights can be loaded using args.pretrained

    Args:
        args: Command line arguments
        config: yaml config.

    Returns: Returns the backbone, bottleneck, and classification networks

    """
    logger = create_logger(output_dir=args.output_dir_src, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    acc_iter = args.acc_iter

    # Use split or all availabel data.
    # Note: For model == all, target dataset must be avaliable.
    # Note : split in image_source.py will use source dataset not the target dataset, please use config.DATA.DATASET to set dataset for fine-tuning with split.
    if args.mode == 'all':
        dataset_train, data_loader_train, mixup_fn = build_loader(config=config,mode=args.mode,dset=config.DATA.DATASET,root= config.DATA.DATA_PATH)

        dataset_val, data_loader_val, _ = build_loader(config,args.mode,target=True,dset=config.DATA.TARGET_DATASET,root= config.DATA.TARGET_DATA_PATH)
    else:
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config=config,mode=args.mode,dset=config.DATA.DATASET,root= config.DATA_PATH)

    dset_loaders = {}
    dset_loaders["source_tr"] = data_loader_train
    dset_loaders["source_te"] = data_loader_val



    
    #Load Feature Extractor 
    pr = True if not config.MODEL.PRETRAINED else False
    print(pr)
    netF = get_backbone(args.net,config,pretrained=pr)
    num_features = netF.num_features
    netF = netF.cuda()
    if config.MODEL.PRETRAINED:
        netF.load_state_dict(torch.load(config.MODEL.PRETRAINED,map_location='cuda:'+str(args.gpu_id)), strict=False)
 

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=num_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=config.MODEL.NUM_CLASSES, bottleneck_dim=args.bottleneck).cuda()


    #Set Learning rate for netF, netB, netC.

    param_group = []
    learning_rate = config.TRAIN.BASE_LR
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 10}]
    
    #AdamW optimizer params initialization:.
    optimizer = optim.AdamW(param_group, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    optimizer = op_copy(optimizer)

    acc_best = 0

    # ForeverDataIter limits 
    N_batch = len(dset_loaders['source_tr'])//acc_iter
    max_iter = config.TRAIN.EPOCHS * len(dset_loaders['source_tr'])
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * len(dset_loaders["source_tr"])) 
    interval_iter = len(dset_loaders['source_tr'])//args.evals_per_epoch

    # Initialize iterating values
    # interval_iter = 10
    iter_num = 0
    epoch_num = 0
    eval_num = 0
    validation_accuracy = []
    
    #Cosuine Scheduler: 
    cosine_lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_iter/acc_iter,
        cycle_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps/acc_iter,
        cycle_limit=1,
        t_in_epochs=False)

    if args.center_loss:
        cent_lr = args.cent_lr
        cent_alpha = args.cent_alpha
        center_loss_func = CenterLoss(num_classes=config.MODEL.NUM_CLASSES, feat_dim=netF.num_features, use_gpu=True)
        optimizer_centloss = torch.optim.AdamW(center_loss_func.parameters(), lr=cent_lr)
    
    # Tensorboard summary writter.
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    
    #Set Network to Training.
    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            print('Starting Epoch Number %d' % epoch_num)
            epoch_num += 1
            running_loss = 0
            running_acc = 0
            tqdm_iter = tqdm(dset_loaders["source_tr"], file=sys.stdout)
            iter_source = iter(tqdm_iter)
            inputs_source, labels_source = next(iter_source)
        
        
        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        # #To use Mixup and Cutmix uncomment line below
        # inputs_source, labels_source = mixup_fn(inputs_source, labels_source)
        # labels_source = torch.argmax(labels_source,dim = 1)

        features = netF(inputs_source)
        outputs_source = netC(netB(features))
        _, preds = torch.max(nn.LogSoftmax(dim=1)(outputs_source.float()), 1)

        classifier_loss = CrossEntropyLabelSmooth(num_classes=config.MODEL.NUM_CLASSES, epsilon=args.smooth)(outputs_source, labels_source)
        classifier_loss = classifier_loss/acc_iter
        if iter_num > warmup_steps and args.center_loss:
               center_loss = center_loss_func(features, labels_source)
               classifier_loss = (cent_alpha*center_loss) + classifier_loss
        classifier_loss.backward()
        

        # Calculating Accuracy and loss of the ongoing training batch
        running_acc +=  (preds == labels_source).float().sum()
        running_loss += classifier_loss.item() * inputs_source.size(0)*acc_iter

        # Batch Accumulation
        if ((iter_num)  % acc_iter == 0) or (iter_num %len(dset_loaders["source_tr"]) == 0):
           if iter_num > warmup_steps and args.center_loss:      
               optimizer_centloss.zero_grad()

           if iter_num > warmup_steps and args.center_loss:
               for param in center_loss_func.parameters():
                   param.grad.data *= (1. / cent_alpha)
               optimizer_centloss.step()
         
           optimizer.step()
           cosine_lr_scheduler.step_update(iter_num)
           optimizer.zero_grad()  

            # Write to tensorboard
           writer.add_scalars(f'LR', {'backbon/step':optimizer.param_groups[0]['lr']}, global_step=iter_num)

        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            

            epoch_loss = running_loss/len(dset_loaders['source_tr'].dataset)
            epoch_acc = 100*running_acc/len(dset_loaders['source_tr'].dataset) 
            netF.eval()
            netB.eval()
            netC.eval()

            # Get Target Accuracy 
            acc_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Aicc_S = {:.2f}%; Acc_T = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_te, acc_te)

            # Write to temsor board.
            writer.add_scalars(f'Loss', {'train':epoch_loss}, global_step=epoch_num)
            tqdm_iter.set_description(log_str)
            writer.add_scalars(f'Accuracy/Source',{'train':epoch_acc,'val_t':acc_te}, global_step=epoch_num)


            logger.info(log_str + '\n')
            # print(log_str + '\n')
            validation_accuracy.append(acc_te)

            if acc_best < acc_te:
                acc_best = acc_te
      
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
            print_top_evals(np.asarray(validation_accuracy), n=TOP_N, logger=logger)
            best_netF = netF.state_dict()
            save_linear_net(best_netF, 'source_F', epoch_num, eval_num, np.asarray(validation_accuracy),
                                args.output_dir_src, top_n=TOP_N)
            save_linear_net(best_netB, 'source_B', epoch_num, eval_num, np.asarray(validation_accuracy),
                            args.output_dir_src, top_n=TOP_N)
            save_linear_net(best_netC, 'source_C', epoch_num, eval_num, np.asarray(validation_accuracy),
                            args.output_dir_src, top_n=TOP_N)
            
            eval_num += 1
      
            netF.train()
            netB.train()
            netC.train()

    log_str = 'Performance for best model: Acc = {:.2f}%'.format(acc_best)
    logger.info(log_str)
    print_top_evals(np.asarray(validation_accuracy), n=TOP_N, logger=logger)

    return netF, netB, netC


def save_linear_net(state_dict, net_name, epoch, eval, accuracies, output_path, top_n=10):
    """
    Save a linear network.  Specifically used for netB and netC in the code.  Removes any output
    that is not in the `TOP_N` results.
    Args:
        state_dict: PyTorch model state_dict
        net_name: Name of the network.  Used for save_name and to remove obsolete save states
        epoch: The epoch number used for save name
        eval: The evaluation number used for save name
        accuracies: List of training accuracy history.  Used to determine if model should be saved as top performer
        output_path: Save directory
        top_n: Number of networks to save

    Returns: None

    """
    save_path = os.path.join(os.path.join(output_path, '%s_epoch_%d_eval_%d.pt' % (net_name, epoch, eval)))
    torch.save(state_dict, save_path)

    # Check if saved checkpoint is no longer in top-10
    top_10_epochs = np.flip(np.argsort(accuracies))[:top_n]
    for file in os.listdir(output_path):
        if net_name in file:
            eval_num = file[file.rfind('_') + 1:file.find('.')]
            eval_num = int(eval_num)
            if eval_num not in top_10_epochs:
                os.remove(os.path.join(output_path, file))


def print_top_evals(validation_accuracy, n=10, logger=None):
    """
    Print the top evaluation results.  Useful for determining performance of saved models
    Args:
        validation_accuracy: List of validation accuracies over training history
        n: The number of top results to print
        logger: Logger object

    Returns: None

    """
    top_n_epochs = np.flip(np.argsort(validation_accuracy))[:n]
    for i in range(top_n_epochs.size):
        epoch_num = top_n_epochs[i]
        epoch_accuracy = validation_accuracy[epoch_num]
        log_str = 'Rank %d: Eval: %d, Acc1: %.3f%%' % (i + 1, epoch_num, epoch_accuracy)
        if logger is not None:
            logger.info(log_str)
        else:
            print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str)
    parser.add_argument('--t-dset', type=str, default='rareplanes-real')
    parser.add_argument('--t-data-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet-50')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--center-loss', default=False)
    parser.add_argument('--cent-lr', default=0.01, type=float)
    parser.add_argument('--cent-alpha', default=0.3, type=float)
    parser.add_argument('--evals-per-epoch', default=10, type=int)
    parser.add_argument('--mode',default='all')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--pretrained',
                        help='manually load weights')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--acc_iter',default = 1,type=int,help='Enter the number of bacth to be accumalted')
    parser.add_argument('--transfer-dataset', action='store_true', help='Transfer the model to a new dataset')
    parser.add_argument('--name', type=str, default='test',
                        help='Unique name for the run')
    parser.add_argument('--lr-test',type=int, default = 0)
    # Args needed to load swin.  Not necessarily used
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    args.eval_period = -1
    print(args.gpu_id)
 

    names = args.net
    config = get_config(args)
            
    torch.distributed.init_process_group(backend='nccl',rank=args.local_rank)

    
    torch.cuda.set_device(int(args.gpu_id))
    set_seed(args.seed)
    # torch.backends.cudnn.deterministic = True


    # Creaye Output directory with model name and store config file
    args.output_dir_src = osp.join(args.output, args.name, names)
    args.name_src = names
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    path = os.path.join(args.output_dir_src, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()


    train_source(args, config)


