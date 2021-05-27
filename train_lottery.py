import sys

import logging
from datetime import datetime
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from transformers import *
import math
import argparse
import random
import copy
import os
from nltk.tokenize import word_tokenize
import pickle

from utils.sst_data_reader import read_sst_data
from utils.utils import evaluate_model, Scheduler
from bert_clf import BertClfModel
from ticket_wrapper import TicketWrapper
from train import complete_train


def save_model(model_save_path, model_state_dict, masks, acc, pp):
    info_to_save = {
        'model_state_dict': model_state_dict, 
        'masks': masks,
    }

    pickle.dump(info_to_save, 
                open(os.path.join(model_save_path,'mask_model_acc{}_density{}.masks'.format(
                    acc, pp)), 'wb'))


if __name__ == '__main__':
    batch_size = 16
    epoch_num = 3 
    checkpoint = False
    gpu = True
    max_grad_norm = 1.
    bert_type = 'bert-base'
    lr = 2e-5

    full_data = read_sst_data('../glue_data/SST-2/train.tsv')
    random.shuffle(full_data)
    train_size = int(len(full_data)*0.95)
    train_data = full_data[:train_size]
    dev_data = full_data[train_size:]
    test_data = read_sst_data('../glue_data/SST-2/dev.tsv')
    print('train data size {}'.format(len(train_data)))
    print('dev data size {}'.format(len(dev_data)))
    print('test data size {}'.format(len(test_data)))

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('batch size:\t{}'.format(batch_size))
    print('epoch num:\t{}'.format(epoch_num))
    print('lr:\t{}'.format(lr))
    print('check_point:\t{}'.format(checkpoint))
    print('gpu:\t{}'.format(gpu))
    print('max grad norm:\t{}'.format(max_grad_norm))
    print('=====Arguments=====')

    base_model = BertClfModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,checkpoint=checkpoint) 
    model = TicketWrapper(base_model, gpu=gpu, not_mask=['pooler','clf_head'])

    # load existing masks
    if len(sys.argv) == 2:
        saved_model_path = sys.argv[1]
        saved_info = pickle.load(open(saved_model_path, 'rb'))
        assert 'masks' in saved_info
        model.load_masks(saved_info['masks'])
        model.load_state_dict(saved_info['model_state_dict'])
        model_save_path = '/'.join(sys.argv[1].split('/')[:-1])
        del saved_info
    else:
        model_save_path = 'output/lottery_{}-{}'.format(bert_type,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
    print('model save path:', model_save_path)

    for prune_step in range(100):
        model.restore_weights()
        pp = model.get_sparsity()
        print('\n===== Prune Step {}, Model Size {} ====='.format(prune_step, pp))
        dev_acc, model_state_dict = complete_train(model, train_data, dev_data, epoch_num, lr, batch_size, gpu, max_grad_norm)
        model.load_state_dict(model_state_dict)
        test_acc = evaluate_model(model, test_data)
        save_model(model_save_path, model_state_dict, model.masks, test_acc, pp) 

        model.update_threshold(model_state_dict)
        model.update_masks(model_state_dict)
        if pp < 1e-4: break
       


