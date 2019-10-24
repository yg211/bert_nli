import sys
sys.path.append('../')
sys.path.append('../apex')

"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from bert_nli import BertNLIModel
from utils.nli_data_reader import NLIDataReader


def evaluate(model, test_data, mute=False):
    model.eval()
    sent_pairs = [test_data[i].get_texts() for i in range(len(test_data))]
    all_labels = [test_data[i].get_label() for i in range(len(test_data))]
    _, probs = model(sent_pairs)
    all_predict = [np.argmax(pp) for pp in probs]
    assert len(all_predict) == len(all_labels)

    acc = len([i for i in range(len(all_labels)) if all_predict[i]==all_labels[i]])*1./len(all_labels)
    prf = precision_recall_fscore_support(all_labels, all_predict, average=None, labels=[0,1,2])

    if not mute:
        print('==>acc<==', acc)
        print('label meanings: 0: contradiction, 1: entail, 2: neutral')
        print('==>precision-recall-f1<==\n', prf)

    return acc


if __name__ == '__main__':
    gpu = True
    batch_size = 16

    # Read the dataset
    nli_reader = NLIDataReader('./datasets/AllNLI')

    model = BertNLIModel(model_path='output/sample_model.state_dict',batch_size=batch_size)
    test_data = nli_reader.get_examples('dev.gz') #,max_examples=50)
    print('test data size: {}'.format(len(test_data)))
    evaluate(model,test_data)
    


