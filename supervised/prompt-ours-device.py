import os
import torch
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging
from domain import get_domainnet_dataset,get_domloader
from OfficeCaltech10 import get_office_caltech10_dloader
from PromptModels.clip_queryv2 import CustomCLIP_server,CustomCLIP_client
import clip
from utils import WarmupCosineSchedule, make_optimizer,evala,evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.5,type=float,)
    parser.add_argument('--data', default='domainnet',help='domainnet')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--round',  default=50, type=int,help='repeat times')
    parser.add_argument('--batch_size', default=256,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--t', default=0.01,type=float,)
    parser.add_argument('--gctx', default=16,type=int,)
    parser.add_argument('--logname', default='oursdevice_gtx16',)
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
        train_datas, test_data = get_domainnet_dataset(args.datapath,domain,args.batch_size,preprocess,args.alpha,5)
        client_dataloaders+=train_datas
        client_testloaders.append(test_data)
    lasttest = get_domloader(domains,args.batch_size*2,preprocess)
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
key = torch.rand([len(domains),args.gctx,512]).cuda()#(16,20)(8,15)

global_model = CustomCLIP_server(out,model,args.gctx,domain_number=len(domains),keys=key,tempr=args.t).cuda()
models  = [CustomCLIP_client(out,model,args.gctx,domain_number=len(domains),keys=key[i//5]).cuda() for i in range(len(client_dataloaders))]
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

for model in models:
    for name, param in model.named_parameters():
        # if "prompt_learner" not in name:
        if 'cquery' not in name and "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
#=======================     Federated training  ==========================================
for fe in tqdm(range(args.round)):
    
    this_round_clients = [np.random.choice(list(range(5*k,5*k+5)), 1, replace=False)[0] for k in range(len(domains))]
    logger.info(f'------------- federated {fe}-th  --------------------------')
    print('------------- federated ',fe,'-th  --------------------------')

    if fe>0:

        for m,client in enumerate(models):
            client.prompt_learner.load_state_dict(global_model.prompt_learner.state_dict(),strict=False)
            client.cquery.load_state_dict(global_model.cquery.state_dict(),strict=False)


    for cl in this_round_clients:
        optimizer = make_optimizer(models[cl].prompt_learner,models[cl].cquery,base_lr=0.01)
        for e in range(1):
            for i, (image, label) in enumerate(tqdm(client_dataloaders[cl])):
                optimizer.zero_grad()
                image = image.to('cuda')
                label = label.to('cuda')
                
                out_dom, out_cls = models[cl](image,True)
                loss = F.cross_entropy(out_cls, label)
                loss += F.cross_entropy(out_dom, label*0+cl//5)
                loss.backward()
                optimizer.step()

  
    weights = [1/len(this_round_clients)]*len(this_round_clients)
    state_global,state_person = [],[]
    
================= aggregation all ================================
    update_query = []
    prompt = [] 
    local_state  = models[0].prompt_learner.state_dict()
    for k, cl in enumerate(this_round_clients):
        client_state = models[cl].prompt_learner.state_dict()
        for st in local_state:
            if k==0:
                local_state[st] = client_state[st]*weights[k]
            else:
                local_state[st] += client_state[st]*weights[k]
            
        update_query.append(models[cl].state_dict()['cquery.weight'].unsqueeze(0))

    global_model.prompt_learner.load_state_dict(local_state,strict=False)
    global_model.load_state_dict({'cquery.weight': torch.mean(torch.cat(update_query,dim=0),dim=0),},strict=False)
#================= aggregation by domain index ================================
    # update_query = [[] for i in range(len(domains))]
    # prompt = [] 
    # local_state  = models[0].prompt_learner.state_dict()
    # for k, cl in enumerate(this_round_clients):
    #     client_state = models[cl].prompt_learner.state_dict()
    #     for st in local_state:
    #         if k==0:
    #             local_state[st] = client_state[st]*weights[k]
    #         else:
    #             local_state[st] += client_state[st]*weights[k]

    #     update_query[cl//5].append(models[cl].state_dict()['cquery.weight'].unsqueeze(0))

    # temp = global_model.state_dict()['cquery.weight']*0
    # alld = 0
    # for do in update_query:
    #     if len(do)==0: 
    #         continue
    #     elif len(do)==1:
    #         temp+=do[0].squeeze(0)
    #         alld+=1
    #     else:
    #         temp+=torch.mean(torch.cat(do,dim=0),dim=0)
    #         alld+=1
    # temp = temp/alld    
    # global_model.prompt_learner.load_state_dict(local_state,strict=False)
    # global_model.load_state_dict({'cquery.weight': temp,},strict=False)

    if fe==args.round-1:
        # torch.save(global_model.prompt_learner, f'./models_newlr/{args.data}_ourdeviceprompt5_{str(fe)}.pth')
        # torch.save(global_model.cquery, f'./models_newlr/{args.data}_ourdevicequery5_{str(fe)}.pth')
        logger.info('epoch: %s' % str(fe))
        for te,test_loader in enumerate(client_testloaders):
            top1,topk = evalaa(global_model,test_loader)
            logger.info('top1: %s' % str(top1))
            print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)        
    

#     