import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import numpy as np
from tqdm import tqdm

from transformers import *
from utils.utils import build_batch


class BertNLIModel(nn.Module):
    """Performs prediction, given the input of BERT embeddings.
    """
    def __init__(self,model_path=None,gpu=True,bert_type='bert-base',label_num=3,batch_size=8,reinit_num=0,freeze_layers=False):
        super(BertNLIModel, self).__init__()
        self.bert_type = bert_type

        if 'bert-base' in bert_type:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif 'bert-large' in bert_type:
            self.bert = BertModel.from_pretrained('bert-large-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        elif 'albert' in bert_type:
            self.bert = AlbertModel.from_pretrained(bert_type)
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_type)
        else:
            print('illegal bert type {}!'.format(bert_type))

        self.num_hidden_layers = self.bert.config.num_hidden_layers
        self.vdim = self.bert.config.hidden_size
        self.nli_head = nn.Linear(self.vdim,label_num)
        self.gpu = gpu
        self.batch_size=batch_size
        self.sm = nn.Softmax(dim=1)
        self.reinit(layer_num=reinit_num, freeze=freeze_layers)

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

    def reinit(self, layer_num, freeze):
        """Reinitialise parameters of last N layers and freeze all others"""
        if freeze:
            for _, pp in self.bert.named_parameters():
                pp.requires_grad = False

        if layer_num >= 0: 
            layer_idx = [self.num_hidden_layers-1-i for i in range(layer_num)]
            layer_names = ['encoder.layer.{}'.format(j) for j in layer_idx]
            for pn, pp in self.bert.named_parameters():
                if any([ln in pn for ln in layer_names]) or 'pooler.' in pn:
                    pp.data = torch.randn(pp.shape)*0.02
                    pp.requires_grad = True


    def load_model(self, sdict):
        if self.gpu:
            self.load_state_dict(sdict)
            self.to('cuda')
        else:
            self.load_state_dict(sdict)

    def forward(self, sent_pair_list, checkpoint=True, bs=None):
        all_probs = None
        if bs is None: 
            bs = self.batch_size
            no_prog_bar = True
        else: no_prog_bar = False
        for batch_idx in tqdm(range(0,len(sent_pair_list),bs), disable=no_prog_bar,desc='evaluate'):
            probs = self.ff(sent_pair_list[batch_idx:batch_idx+bs],checkpoint)[1].data.cpu().numpy()
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


    def step_bert_encode(self, module, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(module.layer):
            if module.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = checkpoint.checkpoint(layer_module, hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if module.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if module.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if module.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if module.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


    def step_checkpoint_bert(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        modules = [module for k, module in self.bert._modules.items()]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = modules[0](input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.step_bert_encode(modules[1], embedding_output,extended_attention_mask,head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = modules[2](sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


    def ff(self,sent_pair_list,checkpoint):
        ids, types, masks = build_batch(self.tokenizer, sent_pair_list, self.bert_type)
        if ids is None: return None
        ids_tensor = torch.tensor(ids) 
        types_tensor = torch.tensor(types) 
        masks_tensor = torch.tensor(masks) 
        
        if self.gpu:
            ids_tensor = ids_tensor.to('cuda')
            types_tensor = types_tensor.to('cuda')
            masks_tensor = masks_tensor.to('cuda')

        if checkpoint:
            cls_vecs = self.step_checkpoint_bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)[1]
        else:
            cls_vecs = self.bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)[1]

        logits = self.nli_head(cls_vecs)
        probs = self.sm(logits)

        # to reduce gpu memory usage
        # del ids_tensor
        # del types_tensor
        # del masks_tensor
        # torch.cuda.empty_cache() # releases all unoccupied cached memory 

        return logits, probs

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
