import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import numpy as np
from tqdm import tqdm
from transformers import *
import random
from torch.optim import SGD
import copy
from nltk.tokenize import word_tokenize
from torch.autograd import Variable
from torch.optim import Optimizer
import pickle


class ModelMasker(nn.Module):
    def __init__(self, ori_model, device='cuda', weights_lr=0.1, mask_lr=1e-3, mask_decay=1e-4):
        super(ModelMasker, self).__init__()
        self.plain_model = copy.deepcopy(ori_model).to(device)
        self.hardt = nn.Hardtanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.masks = None
        self.weights_lr = weights_lr
        self.mask_lr = mask_lr 
        self.mask_decay = mask_decay

        self.cnt = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.hardm = 1e5
        self.softm = 1.
        
        self.mask_weights = {}
        self.update_weights = {}
        self.adam_weights = {}
        for pn, pp in self.plain_model.named_parameters():
            pp.requires_grad = False
            _m = torch.zeros(pp.shape).to(device)
            _v = torch.zeros(pp.shape).to(device)
            self.adam_weights[pn] = {'m':_m, 'v':_v}
            # self.mask_weights[pn] = torch.ones(pp.shape).to(device)*1e-3
            self.mask_weights[pn] = torch.randn(pp.shape).to(device) * 0.03 + 0.1
            self.update_weights[pn] = pp.data
            pp.data = torch.ones(pp.shape).to(device)


    def load_weights(self, maskw, updatew):
        self.mask_weights = {}
        for k in maskw:
            self.mask_weights[k] = maskw[k]
            if 'cuda' in self.device: self.mask_weights[k] = self.mask_weights[k].to(self.device)

    def save(self, spath, info_dict={}):
        info_dict['mask_weights'] = self.mask_weights
        info_dict['b_fnc'] = self.binarise
        with open(spath, 'wb') as fp:
            pickle.dump(info_dict, fp)

            
    '''
    def bin(self,x, thres=0.3):
        return self.sigmoid((self.b(x)-thres)*1e5)
        #return self.hardt(self.relu((x-self.shrink_lambda)*1e5))
    '''

    def binarise(self, x):
        return self.sigmoid(x*self.hardm)


    def sgd_step(self):
        for pn, pp in self.model.named_parameters():
            if pn in self.mask_weights:
                # update mask weights
                soft_bin = self.sigmoid(self.mask_weights[pn]*self.softm).detach()
                gd = self.masks[pn].grad
                self.mask_weights[pn] -= gd*self.mask_lr*soft_bin*(1-soft_bin)
                ones = torch.ones(self.mask_weights[pn].shape).to(self.device)
                self.mask_weights[pn] -= ones*self.mask_decay
                # update weights
                gd = self.update_weights[pn].grad.detach()
                self.update_weights[pn].requires_grad = False
                self.update_weights[pn] -= self.weights_lr * gd 
                # self.update_weights[pn] -= self.weights_lr * gd * soft_bin 
                #self.update_weights[pn] -= self.weights_lr * 0.1 * self.update_weights[pn] # l2 regularization
            
    '''
    def adamw_step(self, mask_lr=1e-3, other_lr=1e-3, mask_decay=1e-3, other_decay=1e-3):
        self.cnt += 1
        for pn, pp in self.model.named_parameters():
            old_m, old_v = self.adam_weights[pn]['m'], self.adam_weights[pn]['v']
            if pn in self.mask_weights:
                soft_bin = self.sigmoid(self.mask_weights[pn]*self.softm).detach()
                m,v,delta = self._adamw_update(old_m, old_v, self.masks[pn].grad*soft_bin*(1-soft_bin))
                #if torch.sum(self.masks[pn].grad).item()**2 > 0.1:
                    #print(pn)
                self.mask_weights[pn] -= delta*mask_lr 
                # sum of weights as regularization
                ones = torch.ones(self.mask_weights[pn].shape).to(self.device)
                self.mask_weights[pn] -= ones*mask_decay
                # negative l2 regularization; purpose: push weights away from zero
                #self.mask_weights[pn] += self.mask_weights[pn].detach()*mask_decay
            self.adam_weights[pn]['m'], self.adam_weights[pn]['v'] = m, v


    def _adamw_update(self, last_m, last_v, grad):
        new_m = self.beta1*last_m + (1-self.beta1)*grad
        new_v = self.beta2*last_v + (1-self.beta2)*grad*grad
        #new_m /= (1.-np.power(self.beta1, self.cnt))
        #new_v /= (1.-np.power(self.beta2, self.cnt))
        delta = new_m/(torch.sqrt(new_v)+self.epsilon)
        return new_m, new_v, delta
    '''
    

    def count_ones(self, verbose=False):
        masks = {}
        pn_stats = {}

        for pn in self.mask_weights: 
            masks[pn] = 1*(self.mask_weights[pn] > 0.)
            # masks[pn] = self.binarise(self.mask_weights[pn]).detach()
        size_cnt = 0.
        ones_cnt = 0.
        for pn in masks:
            sc = np.prod(masks[pn].shape)
            oc = torch.sum(masks[pn]).item()
            size_cnt += sc
            ones_cnt += oc
            if verbose:
                pn_stats[pn] = [oc, sc]

        if not verbose:
            return ones_cnt/size_cnt
        else:
            pn_stats['overall'] = ones_cnt*1./size_cnt
            return pn_stats


    def put_on_masks(self):
        self.masks = {}
        model = copy.deepcopy(self.plain_model)
        for pn, pp in model.named_parameters():
            pp.requires_grad = False
            if pn in self.update_weights:
                self.masks[pn] = Variable(self.binarise(self.mask_weights[pn]).detach()).to(self.device)
                self.masks[pn].requires_grad = True
                self.update_weights[pn].requires_grad = True
                pp *= self.masks[pn]*self.update_weights[pn]
        return model

    def forward(self, inputs):
        self.model = self.put_on_masks()
        return self.model(inputs)




