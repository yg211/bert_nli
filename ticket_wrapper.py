import torch
import torch.nn as nn
import copy
import numpy as np
from collections import OrderedDict

class TicketWrapper():
    def __init__(self, model, device, not_mask=[], prate=0.1):
        # super(TicketWrapper, self).__init__()
        self.model = model
        self.device = device
        pnames = [pn for pn, _ in self.model.named_parameters()]
        self.white_list = []
        for pn in pnames:
            if any([nm in pn for nm in not_mask]):
                continue
            else:
                self.white_list.append(pn)
        self.ori_state_dict = copy.deepcopy(model.state_dict())
        self.masks = self.init_masks()
        self.prate = prate
        self.tvalue = -99
        self.hardtanh = torch.nn.Hardtanh()
        self.relu = torch.nn.ReLU()

    def restore_weights(self):
        self.model.load_state_dict(self.ori_state_dict)
                
    def state_dict(self):
        return self.model.state_dict()

    def parameters(self):
        return self.model.parameters()
    
    def get_masks(self):
        #return copy.deepcopy(self.masks)
        return self.masks
    
    def load_masks(self, masks):
        self.masks = copy.deepcopy(masks)
    
    def load_state_dict(self, model_state):
        self.model.load_state_dict(model_state)
                
    def need_update(self, pname):
        if any( [w==pname  for w in self.white_list] ): return True
        else: return False
                
    def init_masks(self):
        masks = OrderedDict()
        model_state = self.model.state_dict()
        for pn in model_state:
            if not self.need_update(pn): continue
            masks[pn] = torch.ones(model_state[pn].shape).to(self.device)
        return masks
            
    def clear_masks(self):
        self.masks = None 
        
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()
        
    def put_on_masks(self):
        for pn,pp in self.model.named_parameters():
            if not self.need_update(pn): continue
            with torch.no_grad():
                pp *= self.masks[pn]

                
    def __call__(self, *args, **kwargs):
        self.put_on_masks()
        return self.model(*args, **kwargs)
    
    
    def get_sparsity(self):
        model_size = np.sum([np.prod(self.masks[pn].shape) for pn in self.masks])
        link_num = np.sum([torch.sum(self.masks[pn]) for pn in self.masks])
        return link_num/model_size
    

    def update_threshold(self, model_state):
        all_abs_weights = None
        for pn in model_state:
            if not self.need_update(pn): continue
            # flat weights
            masked_pp = model_state[pn]*self.masks[pn] #.cpu()
            abs_flat_pp = torch.abs(masked_pp.view(1,-1))
            if all_abs_weights is None: 
                all_abs_weights = abs_flat_pp
            else:
                all_abs_weights = torch.cat((all_abs_weights,abs_flat_pp), dim =1)

        left_connection_num = np.sum([int(torch.sum(self.masks[mname])) for mname in self.masks])
        k = int(left_connection_num*(1-self.prate))
        tvalue = torch.min( torch.topk(all_abs_weights, k).values )
        self.tvalue = float(tvalue)


    def update_masks(self, model_state):
        for pn in model_state:
            if not self.need_update(pn): continue
            shape = model_state[pn].shape
            # flat weights
            flat_pp = model_state[pn].view(1,-1)
            abs_flat_pp = torch.abs(flat_pp)
            # find weights whose magnitude is smaller than the threshold
            dd = abs_flat_pp-self.tvalue
            new_mask = self.hardtanh( self.relu(dd)*1e5 ).view(shape)
            self.masks[pn] = new_mask.detach().to(self.device)




