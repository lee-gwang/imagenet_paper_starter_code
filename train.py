from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
#from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.utils as util
from utils.utils import print_keep_ratio, GradualWarmupScheduler

import numpy as np
import random
import os, time, sys
import argparse
from model import resnet, resnet_05, resnet_025, resnet_075
from copy import deepcopy
import utils.cg_utils as G

import math
#amp
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#torch.manual_seed(123123)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # True is deterministic
    torch.backends.cudnn.benchmark = True 
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def CrossEntropy(outputs, targets):
    """
    outputs : predict values
    targets : label
    """
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
# model
parser.add_argument('--dataset', '-ds', type=str, default='cifar10', help='dataset')
parser.add_argument('--data_path', '-dp', type=str, default='your dataset path', help='dataset path')
parser.add_argument('--arch_model', '-am', type=str, default='resnet18', help='architecture models')

parser.add_argument('--lr', '-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--scheduler', '-sch', type=str, default='multistep', help='lr_scheduler, [multistep, cosine]')
parser.add_argument('--warmup', '-warmup', action='store_true', help='scheduler warmup')

parser.add_argument('--wt_decay', '-wd', type=float, default=1e-4, help='weight decaying')
parser.add_argument('--epochs', '-e', type=int, default=90, help='epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch size')
parser.add_argument('--gamma',  type=float, default=1e-4, help='threshold related')
parser.add_argument('--label_smoothing', '-lsm', action='store_true', help='label smoothing')
parser.add_argument('--label_smoothing_noise', '-lsm_noise', type=float, default=0.1, help='label smoothing, noise')
# KD
parser.add_argument('--lambda_KD',  type=float, default=0.5, help='KD ratio')

# amp
parser.add_argument('--amp', action='store_true', help='mixed precision')

parser.add_argument('--model_dir', '-md', type=str, default='default', help='saved model path')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--path', '-p', type=str, default=None, help='saved model path(pretrain_load)')

parser.add_argument('--pruned', action='store_true', help='input prune or pretraining')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')

args = parser.parse_args()

#########################
def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

def load_dataset():
    if args.dataset=='imagenet':
        transform_train = transforms.Compose([
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),#ImageNetPolicy(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

        transform_test = transforms.Compose([
                            transforms.Resize(int(224/0.875)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

        # dataset
        trainset = datasets.ImageNet(yourpath, split='train',download=False, transform=transform_train)
        valset = datasets.ImageNet(yourpath, split='train',download=False, transform=transform_test)
        testset = datasets.ImageNet(yourpath, split='val',download=False, transform=transform_test)

        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists(yourpath + 'index_imagenet.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(yourpath + 'index_imagenet.pth')
        # else:
        #     print('!!!!!! Save train_set_index !!!!!!')
        #     torch.save(train_set_index, '/home/leegwang/data/index_imagenet.pth')

        num_sample_valid = 50000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler( train_set_index[:-num_sample_valid]),
                                                num_workers=16, pin_memory=True)
        
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,  
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-num_sample_valid:]),
                                            num_workers=16, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=16 ,pin_memory=True)    

    elif args.dataset=='cifar10':
        # top
        transform_train = transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])


        # dataset
        trainset = torchvision.datasets.CIFAR10(root=yourpath, train=True,download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root=yourpath, train=True,download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root=yourpath, train=False, download=True, transform=transform_test)
        
        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists(yourpath+'index_c10.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(yourpath+'index_c10.pth')
        # else:
        #     print('!!!!!! Save train_set_index !!!!!!')
        #     torch.save(train_set_index, '/home/leegwang/data/index_c10.pth')

        num_sample_valid = 5000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[:-num_sample_valid]),
                                                num_workers=4, pin_memory=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[-num_sample_valid:]),
                                                num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=4, pin_memory=True)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        # dataset
        trainset = torchvision.datasets.CIFAR100(root=yourpath, train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root=yourpath, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root=yourpath, train=False, download=True, transform=transform_test)
        
        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists(yourpath+'index_c100.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(yourpath+'index_c100.pth')
        # else:
        #     print('!!!!!! Save train_set_index !!!!!!')
        #     torch.save(train_set_index, '/home/leegwang/data/index_c100.pth')

        num_sample_valid = 5000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[:-num_sample_valid]),
                                                num_workers=4, pin_memory=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[-num_sample_valid:]),
                                                num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=4, pin_memory=True)
        



    return trainloader, valloader, testloader

#----------------------------
# Define the model.
#----------------------------
def generate_model(model_arch):
    # 0.1
    if model_arch == 'resnet56':
        return resnet.resnet56(dataset=args.dataset)
    elif model_arch == 'resnet50':
        return resnet.resnet50(dataset=args.dataset)
    # 0.75
    if model_arch == 'resnet56_0.75':
        return resnet_075.resnet56(dataset=args.dataset)
    elif model_arch == 'resnet50_0.75':
        return resnet_075.resnet50(dataset=args.dataset)
    # 0.5
    if model_arch == 'resnet56_0.5':
        return resnet_05.resnet56(dataset=args.dataset)
    elif model_arch == 'resnet50_0.5':
        return resnet_05.resnet50(dataset=args.dataset)
    # 0.25
    if model_arch == 'resnet56_0.25':
        return resnet_025.resnet56(dataset=args.dataset)
    elif model_arch == 'resnet50_0.25':
        return resnet_025.resnet50(dataset=args.dataset)
    else:
        raise NotImplementedError("Model architecture is not supported.")

#----------------------------
# Train the network.
#----------------------------


def train_model(net, device, logger):
    # define the loss function
    #seed_everything(42)

    # dataset
    print("Loading the data.")
    trainloader, valloader, testloader = load_dataset()

    #
    if args.label_smoothing:
        criterion = (LabelSmoothingCrossEntropy().cuda() 
                if torch.cuda.is_available() else LabelSmoothingCrossEntropy())
    else:
        criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())
    
    

    if args.dataset=='imagenet':
        lr_decay_milestones = [args.epochs//3, args.epochs//3*2]
    else:
        lr_decay_milestones = [args.epochs//4*2, args.epochs//4*3] # epochs 160 ==>[80, 120]
    
    # initialize the optimizer
    optimizer = optim.SGD(net.parameters(), 
                          lr=0.005 if args.warmup else 0.1, 
                          momentum=0.9, weight_decay = args.wt_decay)

    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer, 
                        milestones=lr_decay_milestones, 
                        gamma=0.1,
                        last_epoch=-1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=args.epochs, 
                        eta_min = 0,
                        last_epoch=-1)

    if args.warmup:
        scheduler2 = GradualWarmupScheduler(optimizer, multiplier=20, total_epoch=5 if args.dataset=='imagenet' else 10, after_scheduler=scheduler)


    #amp
    scaler = GradScaler()
    if args.amp:
        print('mixed precision training!')
    else:
        print('normal training')
    best_acc, best_sparsity = 0 ,0
    #trash
    
    for epoch in range(args.epochs): # loop over the dataset multiple times
        # start training!
        net.train()
        #print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        logger.info(f"current learning rate = {optimizer.param_groups[0]['lr'] :.5f}")
        
        # each epoch
        start = time.time()

        for i, data in enumerate(tqdm(trainloader)):
            if args.amp:
                with autocast():
                    inputs, labels = data[0].to(device), data[1].to(device) # top data

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    
                    # loss
                    loss = torch.FloatTensor([0.]).to(device)
                    loss += criterion(outputs, labels)

                    # amp
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()



            else:
                
                inputs, labels = data[0].to(device), data[1].to(device) # top data
                
                optimizer.zero_grad()
                outputs = net(inputs)
                
                # loss
                loss = torch.FloatTensor([0.]).to(device)
                loss += criterion(outputs, labels)


                loss.backward()
                optimizer.step()


        # update the learning rate
        scheduler2.step() if args.warmup else scheduler.step()

        logger.info('epoch {}'.format(epoch+1))
        acc=  test_accu(valloader, net, device, logger) # original size

        if acc>best_acc:
            best_epoch = epoch+1
            best_acc = acc
            logger.info("Saving the best trained model.")
            torch.save(net.state_dict(), f'{args.model_folder}/best_model.pth')
    
        logger.info(f'학습시간 {time.time()-start : .2f} s')

    logger.info('Finished Training')
    logger.info(f'validation best {best_epoch} epoch, accuracy: {best_acc:.2f}')

    logger.info('############# Test set Accuracy #############')
    net.load_state_dict(torch.load(f'{args.model_folder}/best_model.pth'))
    test_accu(testloader, net, device, logger)


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device, logger):
    correct = 0
    predicted = 0
    total = 0.0


    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))
            
    # accuracy
    logger.info(f'Test ACC : {100*correct/total:.2f}%')
    
    #print_keep_ratio(net, logger)
    return 100*correct/total

#----------------------------
# Main function.
#----------------------------

def main():
    # log
    model_folder = f'./saved_models/' + args.model_dir
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        assert False, 'exist folder name'
    setattr(args, 'model_folder', model_folder)
    logger = util.create_logger(model_folder, 'train', 'info')
    print_args(args, logger)
    ###
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

 

    print("Create {} model.".format(args.arch_model))
    net = generate_model(args.arch_model)
    if args.dataset =='imagenet':
        if len(args.which_gpus)>1:
            net = nn.DataParallel(net)
    elif 'cifar' in args.dataset:
        if len(args.which_gpus)>1:
            net = nn.DataParallel(net)

    net.to(device)
    # train
    print("Start training.")
    train_model(net, device, logger)
    



if __name__ == "__main__":
    seed_everything(42)
    main()
