import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle

from models import *
from complete_train import complete_train
from utils.utils import progress_bar
from ticket_wrapper import TicketWrapper
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

    trainset = CIFAR10(root='./data', train='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader



def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--weight_lr', '-wlr', default=1e-1, type=float, help='weights learning rate')
    parser.add_argument('--epoch_num', '-ep', default=20, type=int, help='epoch num')
    parser.add_argument('--lowest_density', '-ld', default=1e-3, type=float, help='lowest density, between 0 and 1')
    args = parser.parse_args()
    return args.epoch_num, args.weight_lr, args.lowest_density


    
def save_model(model_save_path, model_state_dict, masks, acc, pp):
    info_to_save = {
        'model_state_dict': model_state_dict, 
        'masks': masks,
    }

    pickle.dump(info_to_save, 
            open(os.path.join(model_save_path,'mask_model_acc{:.3f}_density{:.3f}.masks'.format(acc, pp)), 'wb'))


if __name__ == '__main__':

    trainloader, testloader = get_data()
    epoch_num, wlr, lowest_density = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = False

    print('\n===== Arguments =====')
    print('epoch_num:', epoch_num)
    print('weight_lr:', wlr)
    print('lowest density:', lowest_density)
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

    # if 'cuda' in device:
        # net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True

    net = TicketWrapper(net, device)

    for prune_step in range(100):
        net.restore_weights()
        pp = net.get_sparsity()
        print('\n===== Prune Step {}, Density {:.3f} ====='.format(prune_step, pp))
        best_acc, best_weights = complete_train(net, trainloader, testloader, wlr, epoch_num, device)
        net.load_state_dict(best_weights)
        save_model('./imp_weights', best_weights, net.masks, best_acc, pp) 

        net.update_threshold(best_weights)
        net.update_masks(best_weights)
        if pp < lowest_density: break






