import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from complete_train import complete_train
from utils.utils import progress_bar
from mask_model import ModelMasker
from utils.data_loader import CIFAR10


def get_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(
        root='./data', train='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CIFAR10(
        root='./data', train='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader



def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--weight_lr', '-wlr', default=1e-1, type=float, help='weights learning rate')
    parser.add_argument('--mask_lr', '-mlr', default=1e-2, type=float, help='mask learning rate')
    parser.add_argument('--mask_decay', '-mdc', default=1e-4, type=float, help='mask decay rate')
    parser.add_argument('--epoch_num', '-e', default=10, type=int, help='epoch num')
    parser.add_argument('--wanted_density', '-wd', default=1., type=float, help='wanted density, between 0 and 1')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    return args.epoch_num, args.weight_lr, args.mask_lr, args.mask_decay, args.wanted_density, args.resume

if __name__ == '__main__':

    trainloader, testloader = get_data()
    epoch_num, wlr, mlr, mdc, wanted_density, resume = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = True

    print('\n===== Arguments =====')
    print('epoch_num:', epoch_num)
    print('weight_lr:', wlr)
    print('mask_lr:', mlr)
    print('mask_decay:', mdc)
    print('wanted density:', wanted_density)
    print('===== Arguments =====\n')

    print('==> Building model..')
    net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)

    if mask: net = ModelMasker(net, device, wlr, mlr, mdc)

    if 'cuda' in device and not mask:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    complete_train(net, trainloader, testloader, wlr, epoch_num, device, resume, wanted_density, mask=mask)












