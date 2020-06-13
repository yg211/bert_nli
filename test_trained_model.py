import sys
sys.path.append('../')
sys.path.append('../apex')

import torch
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import argparse

from bert_nli import BertNLIModel
from utils.nli_data_reader import NLIDataReader


def evaluate(model, test_data, checkpoint, mute=False, test_bs=10):
    model.eval()
    sent_pairs = [test_data[i].get_texts() for i in range(len(test_data))]
    all_labels = [test_data[i].get_label() for i in range(len(test_data))]
    with torch.no_grad():
        _, probs = model(sent_pairs,checkpoint,bs=test_bs)
    all_predict = [np.argmax(pp) for pp in probs]
    assert len(all_predict) == len(all_labels)

    acc = len([i for i in range(len(all_labels)) if all_predict[i]==all_labels[i]])*1./len(all_labels)
    prf = precision_recall_fscore_support(all_labels, all_predict, average=None, labels=[0,1,2])

    if not mute:
        print('==>acc<==', acc)
        print('label meanings: 0: contradiction, 1: entail, 2: neutral')
        print('==>precision-recall-f1<==\n', prf)

    return acc


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli evaluation")
    ap.add_argument('-b','--batch_size',type=int,default=100,help='batch size')
    ap.add_argument('-g','--gpu',type=int,default=1,help='run the model on gpu (1) or not (0)')
    ap.add_argument('-cp','--checkpoint',type=int,default=0,help='run the model with checkpointing (1) or not (0)')
    ap.add_argument('-tm','--trained_model',type=str,default='default',help='path to the trained model you want to test; if set as "default", it will find in output xx.state_dict, where xx is the bert-type you specified')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-large',help='model you want to test; make sure this is consistent with your trained model')
    ap.add_argument('--hans',type=int,default=0,help='use hans dataset (1) or not (0)')

    args = ap.parse_args()
    return args.batch_size, args.gpu, args.trained_model, args.checkpoint, args.bert_type, args.hans

if __name__ == '__main__':
    batch_size, gpu, mpath, checkpoint, bert_type, hans = parse_args()

    if mpath == 'default': mpath = 'output/{}.state_dict'.format(bert_type)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('trained model path:\t{}'.format(mpath))
    print('gpu:\t{}'.format(gpu))
    print('checkpoint:\t{}'.format(checkpoint))
    print('batch size:\t{}'.format(batch_size))
    print('hans data:\t{}'.format(hans))

    # Read the dataset
    nli_reader = NLIDataReader('./datasets/AllNLI')
    test_data = nli_reader.get_examples('dev.gz') #,max_examples=50)

    if hans:
        nli_reader = NLIDataReader('./datasets/Hans')
        test_data += nli_reader.get_hans_examples('heuristics_evaluation_set.txt')

    model = BertNLIModel(model_path=mpath,batch_size=batch_size,bert_type=bert_type)
    print('test data size: {}'.format(len(test_data)))
    evaluate(model,test_data,checkpoint,test_bs=batch_size)
    


