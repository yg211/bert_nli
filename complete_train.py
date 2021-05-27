'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar


def adjust_decay(net, ones_percent):
    # if len(ones_percent) < 2: return None
    thres = 0.9

    # if ones_percent[-2]-ones_percent[-1]>1e-3: 
        # net.mask_decay *= 0.1
    # elif ones_percent[-2] < ones_percent[-1]: 
        # net.mask_decay * 10
    if ones_percent[-1] < thres:
        net.mask_decay = 0.
        net.mask_lr = 0.


def train(net, trainloader, epoch, device, optimizer, criterion, mask, max_grad_norm=1.):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    ones_percent = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if optimizer is not None: optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if optimizer is not None: optimizer.step()
        else: 
            torch.nn.utils.clip_grad_norm_([net.update_weights[k] for k in net.update_weights], max_grad_norm) 
            torch.nn.utils.clip_grad_norm_([net.mask_weights[k] for k in net.mask_weights], max_grad_norm) 
            net.sgd_step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if mask:
            ones_percent.append(net.count_ones())
            adjust_decay(net, ones_percent)
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f | OnesPercent: %.3f | MaskDecay: %.5f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, ones_percent[-1], net.mask_decay))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net, testloader, epoch,  device, criterion, best_acc, mask):
    # net.eval() # For unclear reasons, turn on evl mode produces random results!
    test_loss = 0
    correct = 0
    total = 0
    ones_percent = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            ones_percent.append(net.count_ones())

            if mask:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | OnesPercent: %.6f'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, np.mean(ones_percent)))
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        return acc 
    else:
        return best_acc

    '''
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if mask:
            state = {
                'mask_weights': net.mask_weights,
                'update_weights': net.update_weights,
                'acc': acc,
                'epoch': epoch,
            }
        else:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return best_acc
    '''


def complete_train(net, trainloader, testloader, resume, lr, epoch_num, device, mask=False):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if mask:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+epoch_num):
        train(net, trainloader, epoch, device, optimizer, criterion, mask)
        best_acc = test(net, testloader, epoch, device, criterion, best_acc, mask)

        if not mask: scheduler.step()
        else: net.weights_lr *= 0.5



