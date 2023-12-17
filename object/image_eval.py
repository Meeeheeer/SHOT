import argparse
import os
import os.path as osp
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
from tqdm import tqdm

import network
from network import get_backbone
import image_target
from utils.config import get_config
from utils.data import build_loader
from utils.logger import create_logger
import matplotlib.colors as col





def viz_tsen(source_features,target_features,output_src):
    """
    Used to get t-SNE plots for two domains. The plot will be stored under the name tsne_viz
    Args:
        source_features : featureas extracted from source domain
        target_features : featureas extracted from target domain
        output_src : output source file
    
    return : none
    """
    source = source_features.numpy()
    target = target_features.numpy()

    domain_labels = np.concatenate((np.ones(len(source)), np.zeros(len(target))))

    features = np.concatenate([source, target], axis=0)
    tsne_dist = TSNE(n_components=2 ,perplexity=40,n_iter=500,random_state=2020).fit_transform(features)


    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(tsne_dist[:, 0], tsne_dist[:, 1], c=domain_labels, cmap=col.ListedColormap(['r', 'b']),labels=['source','target'], s=20)

    plt.legend()
    plt.savefig(output_src+'/tsne_viz.jpg')

def get_features_and_labels(data_loader,netF,netB,netC,use_bottleneck = False):
    """
    Use to extract predictions, label and featureas from netF or netB.
    Args:
        loader: Dataloader
        netF: Backbone network
        netB: Bottleneck network
        netC: Classification network
        use_bottleneck : Flag set to return netB features, False will return features from netF.

    Returns: features, preds, labels
    """
    feats = []
    labels = []
    preds = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs = data[0].cuda()

            #Get Features and predictions
            if use_bottleneck:
                features = netB(netF(inputs))
                pred = netC(features)
            else:
                features = netF(inputs)
                pred = netC(netB(features))

            #Store Features, Predictions, Labels
            labels.append(data[1].cpu())
            preds.append(pred.cpu())
            feats.append(features.cpu())
    
    preds = nn.Softmax(dim=1)(torch.cat(preds,dim=0))
    _ , preds = torch.max(preds,dim=1)

    return torch.cat(feats,dim = 0), torch.cat(labels,dim=0), preds

def get_tsen_viz(args,config):
    """
    Vizualize and T-SNE plots: source against target; for each target , classification report in the output file.
    
    Args:
        args: arguments pass by command line.
        config: yaml config file.
    
    Returns: None
    """
    #Getting Data

    dataset_traget, data_loader_target, _= build_loader(config = config,mode = 'all',target=True, dset=config.DATA.TARGET_DATASET)
    if config.DATA.DATASET:
        dataset_source, data_loader_source, _ = build_loader(config = config,mode = 'all',target=True,dset=config.DATA.DATASET)

    #Setting up model
    netF = get_backbone(args.net,config)
    num_features = netF.num_features
 

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=num_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=config.MODEL.NUM_CLASSES, bottleneck_dim=args.bottleneck).cuda()

    #loading weights
    file_name = config.MODEL.PRETRAINED
    eval_num_str = file_name[file_name.rfind('_') + 1:file_name.find('.')]
    pretrained_dir = os.path.dirname(config.MODEL.PRETRAINED)
    netB_path = args.netB
    netC_path = args.netC
    for file in os.listdir(pretrained_dir):
        if ('eval_%s' % eval_num_str) in file:
            if 'source_B' in file and netB_path == '':
                netB_path = os.path.join(pretrained_dir, file)
            elif 'source_C' in file and netC_path == '':
                netC_path = os.path.join(pretrained_dir, file)
    

    netF.load_state_dict(torch.load(config.MODEL.PRETRAINED,map_location=torch.device('cuda:' + str(args.gpu_id))),strict=False)
    netB.load_state_dict(torch.load(netB_path,map_location=torch.device('cuda:' + str(args.gpu_id))))
    netC.load_state_dict(torch.load(netC_path,map_location=torch.device('cuda:' + str(args.gpu_id))))
    
    #To Cuda!!
    netF.cuda()
    netB.cuda()
    netC.cuda()
    
    #setting to Eval mode
    netF.eval()
    netB.eval()
    netC.eval()

    #Get source and target; features, predictions and true labels
    target_features, target_labels, target_preds = get_features_and_labels(data_loader_target,netF,netB,netC,use_bottleneck=args.use_bottleneck_feaures)

    if config.DATA.DATASET:
        source_features, source_labels, source_preds = get_features_and_labels(data_loader_source,netF,netB,netC,use_bottleneck=args.use_bottleneck_feaures)

    _ = cal_acc(data_loader_target, netF, netB, netC, 'target', out_path=args.output_dir_src, print_out=True) 

    
    if config.DATA.DATASET:
        viz_tsen(source_features,source_labels,target_features,target_labels,args.output_dir_src)
    




def cal_acc(loader, netF, netB, netC, name, eval_psuedo_labels=False, out_path='', print_out=False):
    """
    Evaluation method used to evaluate model performance.  Can also evalute the performance of how the model
    would perform with SHOT-style pseudo labels instead of actual labels.

    Args:
        loader: Dataloader
        netF: Backbone network
        netB: Bottleneck network
        netC: Classification network
        name: Savename for eval
        eval_psuedo_labels: Flag.  If True then the pseudolabels are found and performance on those labels is used.
                            Could be useful for determining how this model would perform during DA training.
        out_path: Save path
        print_out: Flag on whether to print full classification report to the output

    Returns: Model accuracy

    """
    start_test = True

    num_features = netF.num_features
    embeddings = []

    if eval_psuedo_labels:
        mem_label = image_target.obtain_label(loader, netF, netB, netC, args)

    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader))):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            pseudo_idx = data[1]
            inputs = inputs.cuda()
            feat_embeddings = netF(inputs)
            outputs = netC(netB(feat_embeddings))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                if eval_psuedo_labels:
                    all_psuedo = mem_label[pseudo_idx]
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                if eval_psuedo_labels:
                    all_psuedo = np.concatenate((all_psuedo, mem_label[pseudo_idx]), 0)
            embeddings.append(feat_embeddings.detach().cpu().numpy())
    
    embeddings = np.concatenate(embeddings,axis=0)
    all_feats = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_preds = torch.squeeze(predict).float()
    acc = (all_preds == all_label).float().numpy()*100
    # _, all_preds = torch.max(all_output.float(), 1)
    plt.clf()
    cf_matrix = confusion_matrix(all_label, all_preds)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=loader.dataset.classes)
    disp.plot()
    plt.title('CF acc=%.2f%%' % acc.mean())
    plt.savefig(os.path.join(out_path, '%s_cf.png' % name))
    plt.clf()

    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = len(loader.dataset.classes)
    for lab in range(num_categories):
        indices = all_label == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=loader.dataset.classes[lab],
                   alpha=0.5)
    # ax.legend(fontsize='large', markerscale=2)
    plt.title('TSNE acc=%.2f%%' % acc.mean())
    plt.savefig(os.path.join(out_path, '%s_tsne.png' % name))
    plt.clf()

    if eval_psuedo_labels:
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = len(loader.dataset.classes)
        for lab in range(num_categories):
            indices = all_psuedo == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=lab,
                       alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.savefig(os.path.join(out_path, '%s_pseudo_tnse.png' % name))
        plt.clf()

        log_str = classification_report(all_label, all_psuedo, target_names=loader.dataset.classes, digits=4)
        print_all(args.out_file, 'Performance of pseudo labels')
        print_all(args.out_file, log_str)

    # Get the distances
    silhouette_score = metrics.silhouette_score(embeddings, all_label)
    top1_acc = metrics.top_k_accuracy_score(all_label, all_output, k=1)
    top5_acc = metrics.top_k_accuracy_score(all_label, all_output, k=5)

    # class_labels = [int(i) for i in test_loader.dataset.classes]
    log_str = classification_report(all_label, all_preds, target_names=loader.dataset.classes, digits=4)
    if(print_out):
        print_all(args.out_file, 'Performance on: %s' % name)
        print_all(args.out_file, log_str)
        print_all(args.out_file, 'Silhouette Score: %.4f' % silhouette_score)
        print_all(args.out_file, 'Num Parameters: %.3e' % count_parameters(netF))
        print_all(args.out_file, 'top1 acc: %.4f' % top1_acc)
        print_all(args.out_file, 'top5 acc: %.4f' % top5_acc)
        print_all(args.out_file, '------------------------------\n\n')

    plt.close()

    return acc.mean()


def print_all(outfile, string):
    print(string)
    outfile.write(string)
    outfile.flush()




def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--dset', type=str, default='office-home')
    parser.add_argument('--t-dset', type=str, default=None)
    parser.add_argument('--t-data-path', type=str, default=None)
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101, swin-b")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--acc_iter',default=1)
    parser.add_argument('--tr-mode',default=0)
    parser.add_argument('--use_bottleneck_feaures', default=False)
    # Pseudo-label parameters
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)

    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight',default=None)
    parser.add_argument('--netB', default='')
    parser.add_argument('--netC', default='')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--transfer-dataset', action='store_true', help='Transfer the model to a new dataset')
    parser.add_argument('--name', type=str, default='test',
                        help='Unique name for the run')

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
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    args.eval_period = -1

    config = get_config(args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    
    file_name = config.MODEL.PRETRAINED
    eval_num_str = file_name[file_name.rfind('_') + 1:file_name.find('.')]
    args.output_dir_src = osp.join(args.output, 'eval', args.name, 'eval_%s' % eval_num_str)
    # args.name_src = names[args.source][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    path = os.path.join(args.output_dir_src, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    get_tsen_viz(args,config)

