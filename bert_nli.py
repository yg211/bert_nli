import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np

from transformers import BertModel, BertTokenizer
from utils.utils import build_batch


class BertNLIModel(nn.Module):
    """Performs prediction, given the input of BERT embeddings.
    """
    def __init__(self,model_path=None,gpu=True,bert_type='base',label_num=3,batch_size=8):
        super(BertNLIModel, self).__init__()

        if 'base' in bert_type:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.vdim = 768
        else:
            self.bert = BertModel.from_pretrained('bert-large-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.vdim = 1024

        self.nli_head = nn.Linear(self.vdim,label_num)
        self.gpu = gpu
        self.batch_size=batch_size

        # load trained model
        if model_path is not None:
            if gpu:
                sdict = torch.load(model_path)
                self.load_state_dict(sdict)
                self.to('cuda')
            else:
                sdict = torch.load(model_path,map_location=lambda storage, loc: storage)
                self.load_state_dict(sdict)
        else:
            if self.gpu: self.to('cuda')

    def load_model(self, sdict):
        if self.gpu:
            self.load_state_dict(sdict)
            self.to('cuda')
        else:
            self.load_state_dict(sdict)

    def forward(self, sent_pair_list):
        all_probs = None
        for batch_idx in range(0,len(sent_pair_list),self.batch_size):
            probs = self.ff(sent_pair_list[batch_idx:batch_idx+self.batch_size]).data.cpu().numpy()
            if all_probs is None:
                all_probs = probs
            else:
                all_probs = np.append(all_probs, probs, axis=0)
        labels = []
        for pp in all_probs:
            ll = np.argmax(pp)
            if ll==0:
                labels.append('contradiction')
            elif ll==1:
                labels.append('entail')
            else:
                assert ll==2
                labels.append('neutral')
        return labels, all_probs

    def ff(self,sent_pair_list):
        ids, types, masks = build_batch(self.tokenizer, sent_pair_list)
        if ids is None: return None
        ids_tensor = torch.tensor(ids)
        types_tensor = torch.tensor(types)
        masks_tensor = torch.tensor(masks)
        
        if self.gpu:
            ids_tensor = ids_tensor.to('cuda')
            types_tensor = types_tensor.to('cuda')
            masks_tensor = masks_tensor.to('cuda')
            #self.bert.to('cuda')
            #self.nli_head.to('cuda')

        cls_vecs = self.bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)[1]
        logits = self.nli_head(cls_vecs)
        predict_probs = F.log_softmax(logits,dim=1)
        return predict_probs

    def save(self, output_path, config_dic=None, acc=None):
        if acc is None:
            model_name = 'nli_model.state_dict'
        else:
            model_name = 'nli_model_acc{}.state_dict'.format(acc)
        opath = os.path.join(output_path, model_name)
        if config_dic is None:
            torch.save(self.state_dict(),opath)
        else:
            torch.save(config_dic,opath)

    @staticmethod
    def load(input_path,gpu=True,bert_type='base',label_num=3,batch_size=16):
        if gpu:
            sdict = torch.load(input_path)
        else:
            sdict = torch.load(input_path,map_location=lambda storage, loc: storage)
        model = BertNLIModel(gpu,bert_type,label_num,batch_size)
        model.load_state_dict(sdict)
        return model
