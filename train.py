import sys
sys.path.append('./apex')

"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset with softmax loss function. 
At every 1000 training steps, the model is evaluated on the dev set.
"""
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

from utils.nli_data_reader import NLIDataReader
from utils.logging_handler import LoggingHandler
from bert_nli import BertNLIModel
from test_trained_model import evaluate


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler=='constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler=='warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler=='warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_acc):
    loss_fn = nn.CrossEntropyLoss()

    step_cnt = 0
    best_model_weights = None

    for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
        model.train() # model was in eval mode in evaluate(); re-activate the train mode
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory 

        step_cnt += 1
        sent_pairs = []
        labels = []
        for i in range(pointer, pointer+batch_size):
            if i >= len(train_data): break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300: continue
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        logits, _ = model.ff(sent_pairs,checkpoint)
        if logits is None: continue
        true_labels = torch.LongTensor(labels)
        if gpu:
            true_labels = true_labels.to('cuda')
        loss = loss_fn(logits, true_labels)

        # back propagate
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update weights 
        optimizer.step()

        # update training rate
        scheduler.step()

        if step_cnt%2000 == 0:
            acc = evaluate(model,dev_data,checkpoint,mute=True)
            logging.info('==> step {} dev acc: {}'.format(step_cnt,acc))
            if acc > best_acc:
                best_acc = acc
                best_model_weights = copy.deepcopy(model.cpu().state_dict())
                model.to('cuda')

    return best_model_weights


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('-b','--batch_size',type=int,default=8,help='batch size')
    ap.add_argument('-ep','--epoch_num',type=int,default=1,help='epoch num')
    ap.add_argument('--fp16',type=int,default=0,help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
    ap.add_argument('--check_point','-cp',type=int,default=1,help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use this together with apex fp16')
    ap.add_argument('--gpu',type=int,default=1,help='use gpu (1) or not (0)')
    ap.add_argument('-tr','--train_rate',type=float,default=0.95,help='how many data are used in training (the rest will be used as dev)')
    ap.add_argument('-ss','--scheduler_setting',type=str,default='WarmupLinear',choices=['WarmupLinear','ConstantLR','WarmupConstant','WarmupCosine','WarmupCosineWithHardRestarts'])
    ap.add_argument('-tm','--trained_model',type=str,default='None',help='path to the trained model; make sure the trained model is consistent with the model you want to train')
    ap.add_argument('-mg','--max_grad_norm',type=float,default=1.,help='maximum gradient norm')
    ap.add_argument('-wp','--warmup_percent',type=float,default=0.2,help='how many percentage of steps are used for warmup')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-large',help='transformer (bert) pre-trained model you want to use', choices=['bert-base','bert-large','albert-base-v2','albert-large-v2'])

    args = ap.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu, args.train_rate, args.scheduler_setting, args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model


if __name__ == '__main__':

    batch_size, epoch_num, fp16, checkpoint, gpu, train_rate, scheduler_setting, max_grad_norm, warmup_percent, bert_type, trained_model = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)
    checkpoint = bool(checkpoint)
    if trained_model=='None': trained_model=None

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('trained model path:\t{}'.format(trained_model))
    print('batch size:\t{}'.format(batch_size))
    print('epoch num:\t{}'.format(epoch_num))
    print('fp16:\t{}'.format(fp16))
    print('check_point:\t{}'.format(checkpoint))
    print('gpu:\t{}'.format(gpu))
    print('train rate:\t{}'.format(train_rate))
    print('scheduler setting:\t{}'.format(scheduler_setting))
    print('max grad norm:\t{}'.format(max_grad_norm))
    print('warmup percent:\t{}'.format(warmup_percent))
    print('=====Arguments=====')

    label_num = 3
    model_save_path = 'output/nli_{}-{}'.format(bert_type,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print('model save path', model_save_path)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # Read the dataset
    nli_reader = NLIDataReader('datasets/AllNLI')
    train_num_labels = nli_reader.get_num_labels()

    all_data = nli_reader.get_examples('train.gz') #,max_examples=5000)
    random.shuffle(all_data)
    train_data = all_data[:int(train_rate*len(all_data))]
    dev_data = all_data[int(train_rate*len(all_data)):]

    logging.info('train data size {}'.format(len(train_data)))
    logging.info('dev data size {}'.format(len(dev_data)))
    total_steps = math.ceil(epoch_num*len(train_data)*1./batch_size)
    warmup_steps = int(total_steps*warmup_percent)

    model = BertNLIModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,model_path=trained_model) 
    optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-6,correct_bias=False)
    scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    best_acc = -1.
    best_model_dic = None
    for ep in range(epoch_num):
        logging.info('\n=====epoch {}/{}====='.format(ep,epoch_num))
        model_dic = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_acc)
        if model_dic is not None:
            best_model_dic = model_dic
    assert best_model_dic is not None

    # for testing load the best model
    model.load_model(best_model_dic)
    logging.info('\n=====Training finished. Now start test=====')
    test_data = nli_reader.get_examples('dev.gz') #,max_examples=50)
    logging.info('test data size: {}'.format(len(test_data)))
    test_acc = evaluate(model,test_data,batch_size)
    logging.info('accuracy on test set: {}'.format(test_acc))

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        if os.listdir(model_save_path):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(
                model_save_path))
    model.save(model_save_path,best_model_dic,test_acc)
