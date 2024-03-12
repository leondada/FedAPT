import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging
import timm
import math
from domain import get_domainnet_dloader
from OfficeCaltech10 import get_office_caltech10_dloader
from PromptModels.clip_queryv2 import CustomCLIP_server,CustomCLIP_client
import clip
from utils import make_optimizer,evala,evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='domainnet',help='domainnet or office_caltech10')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--round',  default=50, type=int,help='repeat times')
    parser.add_argument('--batch_size', default=256,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--t', default=0.01,type=float,help='when t->0, means one-hot')
    parser.add_argument('--gctx', default=24,type=int,help='the length of prompts')
    parser.add_argument('--logname', default='ours_gtx16',)
    parser.add_argument('--datapath', default='...',)
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
    filename=f'./logfinal/{args.data}_{args.logname}.log',
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
        train_data, test_data = get_domainnet_dloader(args.datapath,domain,args.batch_size,preprocess)
        client_datasets.append(train_data)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size*2, num_workers=8,shuffle=False, pin_memory=True))
        
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



#=======================     Initialize keys and models  ==========================================
key = torch.rand([len(domains),args.gctx,512]).cuda()#(16,20)(8,15)
global_model = CustomCLIP_server(out,model,args.gctx,domain_number=len(domains),keys=key,tempr=args.t).cuda()
models  = [CustomCLIP_client(out,model,args.gctx,domain_number=len(domains),keys=key[i]).cuda() for i in range(len(client_dataloaders))]
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())


for model in models:
    for name, param in model.named_parameters():
        if 'cquery' not in name and "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

#=======================     Federated training  ==========================================
for fe in range(args.round):
    logger.info(f'------------- federated {fe}-th  --------------------------')
    print('------------- federated ',fe,'-th  --------------------------')
    

    if fe>0:
        for m,client in enumerate(models):
            client.prompt_learner.load_state_dict(global_model.prompt_learner.state_dict(),strict=False)
            client.cquery.load_state_dict(global_model.cquery.state_dict(),strict=False)
        
        
    
    for cl,client in enumerate(tqdm(models)):
        optimizer = make_optimizer(client.prompt_learner,client.cquery,base_lr=0.01)
        for e in range(1):
            for i, (image, label) in enumerate(tqdm(client_dataloaders[cl])):
                optimizer.zero_grad()
                image = image.to('cuda')
                label = label.to('cuda')
                out_dom, out_cls = client(image,True)
                loss = F.cross_entropy(out_cls, label)
                loss += F.cross_entropy(out_dom, label*0+cl)
                loss.backward()
                optimizer.step()

            
    weights = [1/len(models)]*len(models)
    update_query = []

    prompt = [] 
    local_state  = models[0].prompt_learner.state_dict()
    for k, client in enumerate(models):
        client_state = client.prompt_learner.state_dict()
        for st in local_state:
            if k==0:
                local_state[st] = client_state[st]*weights[k]
            else:
                local_state[st] += client_state[st]*weights[k]
            
        update_query.append(client.state_dict()['cquery.weight'].unsqueeze(0))

    global_model.prompt_learner.load_state_dict(local_state,strict=False)
    global_model.load_state_dict({'cquery.weight': torch.mean(torch.cat(update_query,dim=0),dim=0),},strict=False)

    # torch.save(global_model.prompt_learner, f'./models_newlr/{args.data}_randkeyprompt_{str(fe)}.pth')
    # torch.save(global_model.cquery, f'./models_newlr/{args.data}_randkeyquery_{str(fe)}.pth')
    if fe==args.round-1:
        logger.info('epoch: %s' % str(fe))
        for te,test_loader in enumerate(client_testloaders):
            top1,topk = evalaa(global_model,test_loader)
            logger.info('top1: %s' % str(top1))
            print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)        
    

#     