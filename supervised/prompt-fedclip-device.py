import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging
from domain import get_domainnet_dataset
from OfficeCaltech10 import get_office_caltech10_dloader
from PromptModels.clip_base import CustomCLIP_ad
import clip
from utils import make_optimizer,evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.5,type=float,)
    parser.add_argument('--data', default='domainnet',help='domainnet')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--round',  default=30, type=int,help='repeat times')
    parser.add_argument('--batch_size', default=256,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--gctx', default=0,type=int,)
    parser.add_argument('--logname', default='basedevice_gtx16',)
    parser.add_argument('--datapath', default='/home/share/DomainNet',)
    parser.add_argument('--choose', default='rand',)
    return parser


parser = get_parser()
args = parser.parse_args()


def setup_seed(seed):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)


import logging
logging.basicConfig(
    filename=f'./logfinal/{args.choose}_{args.logname}_{args.alpha}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(torch.device('cuda'))


    

    
model, preprocess = clip.load("ViT-B/32", device='cuda')


#======================= dataset ==========================================
if args.data == 'domainnet':
    numclass=345
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']#['clipart','quickdraw',] #
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_datas, test_data = get_domainnet_dataset(args.datapath,domain,args.batch_size,preprocess,args.alpha,5)
        client_dataloaders+=train_datas
        client_testloaders.append(test_data)
        
    out = ['None']*345
    with open('/home/share/DomainNet/splits/clipart_test.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            out[int(label)] = data_path.split('/')[1]
    out  = [name.replace("_", " ") for name in out]

elif args.data == 'office_caltech10':
    numclass=10
    domains = ['amazon', 'webcam', 'dslr', "caltech"]
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data, test_data ,strain,stest= get_office_caltech10_dloader(args.datapath,domain,args.batch_size,preprocess)
        client_datasets.append(train_data)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=strain))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=stest))
    
    out = ['back pack','bike','calculator','headphones','keyboard','laptop computer','monitor','mouse','mug','projector']

#=======================  Initialize keys and models   ==========================================

global_model = CustomCLIP_ad(out,model,0,domain_number=len(domains)).cuda()
models  = [CustomCLIP_ad(out,model,0,domain_number=len(domains)).cuda() for i in range(len(client_dataloaders))]
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

for model in models:
    for name, param in model.named_parameters():
        if "ad_last" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

#=======================     Federated training  ==========================================         
for fe in tqdm(range(args.round)):
    if args.choose=='rand':
        this_round_clients = np.sort(np.random.choice(list(range(30)), 6, replace=False)).tolist()
        # [np.random.choice(list(range(5*k,5*k+5)), 1, replace=False)[0] for k in range(len(domains))]
    else:
        this_round_clients = [np.random.choice(list(range(5*k,5*k+5)), 1, replace=False)[0] for k in range(len(domains))]
    print(this_round_clients)
    logger.info(f'------------- federated {fe}-th  --------------------------')
    print('------------- federated ',fe,'-th  --------------------------')

    if fe>0:
        for m,client in enumerate(models):
            client.ad_last.load_state_dict(global_model.ad_last.state_dict(),strict=False)
        
    for cl in this_round_clients:
        optimizer = make_optimizer(models[cl].ad_last,base_lr=0.01)
        for e in range(1):
            for i, (image, label) in enumerate(tqdm(client_dataloaders[cl])):
                optimizer.zero_grad()
                image = image.to('cuda')
                label = label.to('cuda')
                out_cls = models[cl](image,True)
                loss = F.cross_entropy(out_cls, label)
                loss.backward()
                optimizer.step()
            
    weights = [1/len(this_round_clients)]*len(this_round_clients)


    prompt = [] 
    local_state  = models[0].ad_last.state_dict()
    for k, cl in enumerate(this_round_clients):
        client_state = models[cl].ad_last.state_dict()
        for st in local_state:
            if k==0:
                local_state[st] = client_state[st]*weights[k]
            else:
                local_state[st] += client_state[st]*weights[k]
        

    global_model.ad_last.load_state_dict(local_state,strict=False)

    if fe==args.round-1:
        logger.info('epoch: %s' % str(fe))
        for te,test_loader in enumerate(client_testloaders):
            top1,topk = evalaa(global_model,test_loader)
            logger.info('top1: %s' % str(top1))
            print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)        
    

#     