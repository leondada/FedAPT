
import torch
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from tqdm import tqdm
#======================= utils ==========================================


def evalaa(model, testdata):
    # model.eval()
    # i = 0
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(testdata):
            # i+=1
            # if i>2: break
            test_labels = test_labels.cuda()
            out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
    # print(top1,topk,total)
    # model.train()
    return 100 * top1 / total,100*topk/total

def evala(model, testdata):
    # model.eval()
    # i = 0
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(testdata):
            # i+=1
            # if i>2: break
            test_labels = test_labels.cuda()
            _,out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
    # print(top1,topk,total)
    # model.train()
    return 100 * top1 / total,100*topk/total



def lr_cos(step):
        # if step < 5:
        #     return float(step) / float(max(1.0, 5))
        # progress after warmup
        progress = float(step) / float(max(
            1, 50))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * 0.5 * 2.0 * progress))
        )
    
    
def make_optimizer(model,model2=None,base_lr=0.002,iround=1,WEIGHT_DECAY=1e-5):
    params = []
    # only include learnable params
    for key, value in model.named_parameters():
        if value.requires_grad:
            params.append((key, value))
    if model2!=None:
        for key, value in model2.named_parameters():
            if value.requires_grad:
                params.append((key, value))


    _params = []
    for p in params:
        key, value = p
        # print(key)
        # if not value.requires_grad:
        #     continue
        if 'cquery' in key:
            tlr = base_lr#*lr_cos(iround)
        else:
            tlr = base_lr
        weight_decay = WEIGHT_DECAY
        _params += [{
            "params": [value],
            "lr": tlr,
            "weight_decay": weight_decay
        }]

    optimizer = torch.optim.SGD(
        _params,lr = base_lr,momentum=0.9,weight_decay=WEIGHT_DECAY
    )
    return optimizer