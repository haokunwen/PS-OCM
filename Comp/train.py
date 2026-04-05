import argparse
import os
import sys
import json
import logging
import warnings
from tqdm import tqdm

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from dataset import IQON_dataset
from torch.utils.data import dataloader
import model

from torch.cuda.amp import autocast as autocast, GradScaler

torch.set_num_threads(4)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PS-OCM')
parser.add_argument('--datadir', type=str, default='../data/',
                    help='directory of the IQON3000 outfits dataset')
parser.add_argument('--imgpath', type=str, default='/home/share/wangchun/KGAT-pytorch-master/other/IQON3000/IQON3000/')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epoch_num', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--embedding_dim', type=int, default=256)

parser.add_argument('--max_outfit', type=int, default=10)
parser.add_argument('--model_dir', type=str, default='./result')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--ortho', type=float, default=1.0)
parser.add_argument('--deconv', type=float, default=1.0)
parser.add_argument('--partial', type=float, default=1.0)


args = parser.parse_args()

def load_dataset(args):


    trainset = IQON_dataset(args, split='train', 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.RandomCrop(224),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    validset = IQON_dataset(args, split='valid', 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    testset = IQON_dataset(args, split='test',
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    print('trainset size:', len(trainset))
    print('valid size:', len(validset))
    print('testset size:', len(testset))

    return trainset, validset, testset


def create_model_and_optimizer(att_num_dic):
    """Builds the model and related optimizer."""

    ps_ocm = model.Image_net(embedding_dim=args.embedding_dim, outfit_threshold = args.max_outfit,att_num_dic= att_num_dic).cuda()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, ps_ocm.parameters()),lr = args.lr, eps=args.eps, weight_decay=args.weight_decay)

    return ps_ocm, optimizer

def compute_kl(x1, x2):
    log_soft_x1 = F.log_softmax(x1, dim=1)
    soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
    kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
    return kl

def compute_mse(x1, x2):
    return F.mse_loss(x1, x2)

def compute_auc_acc(predicted, label):

    fpr, tpr, thresholds = metrics.roc_curve(label, predicted, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)

    acc_predicted_score = predicted.squeeze()
    ones = np.ones_like(acc_predicted_score)
    zeros = np.zeros_like(acc_predicted_score)
    acc_predicted_score = np.where(acc_predicted_score > 0.5, ones, zeros)

    acc_score = metrics.accuracy_score(label, acc_predicted_score)
    return auc_score, acc_score

def train(model, optimizer, dataloader, scaler):
    model.train()

    img_all_predicted_socre = []
    all_labels = []

    with tqdm(total = len(dataloader)) as t:
        for i,data in enumerate(dataloader):

            img = [t['img'] for t in data]
            att_label = [t['att_label'] for t in data]  
            att_mask = [t['att_mask'] for t in data]  
            target = [t['target'] for t in data]
            partial_mask = [t['partial_mask'] for t in data]

            att_mask = torch.tensor(att_mask).unsqueeze(3).cuda()  #(batch_size,item_num,att_num) 
            att_label = torch.tensor(att_label).cuda() 
            partial_mask = torch.tensor(partial_mask).unsqueeze(3).cuda()

            labels = torch.tensor(target).squeeze(1).cuda()

            optimizer.zero_grad()
            with autocast(enabled=False):
                img_predicted_score,partial_supervision_loss,loss_r2,ortho_loss_r1 = model(img,att_mask,att_label,partial_mask)

            img_classify_loss = F.cross_entropy(img_predicted_score, labels)
            img_total_loss = img_classify_loss + args.partial * partial_supervision_loss + args.deconv * loss_r2 + args.ortho * ortho_loss_r1
               
            scaler.scale(img_total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            predicted_y = F.softmax(img_predicted_score, dim=1)
            predicted_y = predicted_y[:,1]
            img_all_predicted_socre += [predicted_y.data.cpu().numpy()]
            all_labels += [labels.data.cpu().numpy()]
            t.set_postfix(img_classify_loss='{:05.3f}'.format(img_classify_loss.item()))
            t.update()

def test(model, dataloader):
    model.eval()

    img_all_predicted_socre = []
    all_labels = []
    with torch.no_grad():
        with tqdm(total = len(dataloader)) as t:
            for i,data in enumerate(dataloader):
                img = [t['img'] for t in data]
                att_label = [t['att_label'] for t in data]
                att_mask = [t['att_mask'] for t in data]
                target = [t['target'] for t in data]
                partial_mask = [t['partial_mask'] for t in data]
                
                att_mask = torch.tensor(att_mask).unsqueeze(3).cuda() 
                att_label = torch.tensor(att_label).cuda()
                partial_mask = torch.tensor(partial_mask).unsqueeze(3).cuda()
                
                img_predicted_score = model(img,att_mask,att_label,partial_mask)[0] 

                labels = torch.tensor(target).squeeze(1).cuda()
                predicted_y = F.softmax(img_predicted_score, dim=1)
                predicted_y = predicted_y[:,1]
                img_all_predicted_socre += [predicted_y.data.cpu().numpy()]
                all_labels += [labels.data.cpu().numpy()]

                t.update()
    img_all_predicted_socre = np.concatenate(img_all_predicted_socre)

    all_labels = np.concatenate(all_labels)
    avg_auc, avg_acc = compute_auc_acc(img_all_predicted_socre, all_labels)

    return avg_auc, avg_acc

def train_and_evaluate(model, optimizer, trainset, validset, testset):

    
    trainloader = dataloader.DataLoader(trainset, 
                             batch_size = args.batch_size,
                             shuffle = True,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn=lambda i:i)
    validloader = dataloader.DataLoader(validset, 
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn = lambda i:i)      
    testloader = dataloader.DataLoader(testset, 
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn= lambda i :i)    
    scaler = GradScaler()

    best_avg_score = float('-inf')
    for epoch in range(args.epoch_num):

        logging.info("Epoch {}/{}".format(epoch + 1, args.epoch_num))
        train(model, optimizer, trainloader, scaler)

        valid_avg_auc, valid_avg_acc = test(model, validloader)
        logging.info("valid avg auc {}, valid avg acc {}".format(valid_avg_auc, valid_avg_acc))

        if valid_avg_auc > best_avg_score:
            best_avg_score = valid_avg_auc
            test_avg_auc, test_avg_acc = test(model, testloader)
            logging.info("Fine new avg best score at epoch {}, test_avg_auc {}, test_avg_acc {}".format(epoch, test_avg_auc, test_avg_acc))
            best_test_saved_avg = os.path.join(args.model_dir, "metrics_best_avg.txt")
            with open(best_test_saved_avg, 'w') as f:
                f.write('%s: test_avg_auc: %s  test_avg_acc: %s ' % (str(epoch), str(test_avg_auc), str(test_avg_acc)))
            torch.save(model, os.path.join(args.model_dir, 'model.pt'))
           


if __name__ == '__main__':

    # Load the parameters from json file

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(os.path.join(args.model_dir, 'train.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    trainset, validset, testset = load_dataset(args)
    att_num_dic = trainset.att_num_dic
    model, optimizer = create_model_and_optimizer(att_num_dic)

    logging.info('- done.')

    # Train the model
    logging.info("Starting train for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, optimizer, trainset, validset, testset)
