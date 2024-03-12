import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import torch
import torch.nn as nn
import torchvision
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
import math
from domain import get_domainnet_dloader,get_domloader,get_domainnet_dataset
from OfficeCaltech10 import get_office_caltech10_dloader
from PromptModels.clip_queryv2 import CustomCLIP_server, CustomCLIP_client,visual,CustomCLIP_temp
import clip
from utils import make_optimizer,evala
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import time

_tokenizer = _Tokenizer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.5,type=float,)
    parser.add_argument('--data', default='domainnet',help='only for domainnet')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--round',  default=50, type=int)
    parser.add_argument('--batch_size', default=512,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--gctx', default=16,type=int,)
    parser.add_argument('--t', default=0.01,type=float,help='when t->0, means one-hot')
    parser.add_argument('--csc', default=False,type=bool,)
    parser.add_argument('--logname', default='ceshi',)
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
    filename=f'./logfinal/ours_{args.alpha}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(torch.device('cuda'))


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class MultiViewDataInjector(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample,):
        output = [transform(sample) for transform in self.transforms]
        return output

    
    
DEFAULT_AUG = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],p = 0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))],p = 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711])),]
        )

    
clip_model, precess = clip.load("ViT-B/32", device='cuda')

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224)])
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])
preprocess = transforms.Compose([transforms.ToTensor(),normalize])


batchsize = 1

#======================= dataset ==========================================
if args.data == 'domainnet' and args.alpha<0:
    numclass=345
    domains = ['clipart', 'infograph', 'painting','quickdraw', 'real', 'sketch']
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    lens = [0,0,0,0,0,0]
    for i,domain in enumerate(domains):
        _, test_data = get_domainnet_dloader(domain,args.batch_size,MultiViewDataInjector([precess,DEFAULT_AUG,DEFAULT_AUG]))#
        client_dataloaders.append(torch.utils.data.DataLoader(test_data,batch_size=args.batch_size, shuffle=True,num_workers=8, pin_memory=True))
        
        _, test_data2 = get_domainnet_dloader(domain,args.batch_size,MultiViewDataInjector([precess,]))
        client_testloaders.append(torch.utils.data.DataLoader(test_data2,batch_size=args.batch_size, shuffle=False,num_workers=8, pin_memory=True))
        
        lens[i] = test_data.__len__()
        
    out = ['None']*345
    with open('/home/share/DomainNet/splits/clipart_test.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            out[int(label)] = data_path.split('/')[1]
    out  = [name.replace("_", " ") for name in out]
    
elif args.data == 'domainnet' and args.alpha>0:
    numclass=345
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']#['clipart','quickdraw',] #
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    lens = []
    for i,domain in enumerate(domains):
        train_datas, test_data ,lensi= get_domainnet_dataset(domain,args.batch_size,MultiViewDataInjector([precess,DEFAULT_AUG,DEFAULT_AUG]),MultiViewDataInjector([precess,]),args.alpha,5)
        client_dataloaders+=train_datas
        client_testloaders.append(test_data)
        lens+=lensi
        
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
        train_data, test_data ,strain,stest= get_office_caltech10_dloader(domain,args.batch_size,preprocess)
        client_datasets.append(train_data)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=strain))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=stest))
    
    out = ['back pack','bike','calculator','headphones','keyboard','laptop computer','monitor','mouse','mug','projector']

#=======================  Initialize keys and models   ==========================================
print(args.csc)
key = torch.rand([len(domains),args.gctx,512]).cuda()
tempmodel = CustomCLIP_temp(out,clip_model,0,CSC=args.csc).cuda()

visualmodel = visual(clip_model).cuda()
# global_model = CustomCLIP_client(out,clip_model,args.gctx,CSC=args.csc).cuda()
models = []

global_model = CustomCLIP_server(out,clip_model,args.gctx,domain_number=len(domains),CSC=args.csc,keys=key,tempr=args.t).cuda()
if args.alpha<0:
    models  = [CustomCLIP_client(out,clip_model,args.gctx,domain_number=len(domains),keys=key[i],CSC=args.csc).cuda() for i in range(len(client_dataloaders))]
else:
    models  = [CustomCLIP_client(out,clip_model,args.gctx,domain_number=len(domains),keys=key[i//5],CSC=args.csc).cuda() for i in range(len(client_dataloaders))]

for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

for name, param in visualmodel.named_parameters():
    param.requires_grad_(False)


for model in models:
    for name, param in model.named_parameters():
        if "prompt_learner" not in name and 'cquery' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

#=======================     Federated training  ==========================================

print(lens)
def lrcos(step=0,lr=0.01,lr_min=0.0001,T_max=500):
    return 0.5*(1 + math.cos(math.pi * step / T_max)) *(lr - lr_min) + lr_min


testlodd = get_domloader(domains, 512, precess, num_workers=16)
            
for gf in range(1):

    if gf>0: 
        tempmodel = copy.deepcopy(models)
    if gf==0:
        for fe in range(10):
            if args.alpha<0:
                this_round_clients = list(range(6))
            else:
                this_round_clients = [np.random.choice(list(range(5*k,5*k+5)), 1, replace=False)[0] for k in range(len(domains))]
                
            # print(this_round_clients)
            for m,client in enumerate(models):
                client.prompt_learner.load_state_dict(global_model.prompt_learner.state_dict(),strict=False)
                client.cquery.load_state_dict(global_model.cquery.state_dict(),strict=False)

            logger.info(f'------------- federated {fe}-th  --------------------------')
            print('------------- federated ',fe,'-th  --------------------------')

            alpha0 = 1
            beta0 = 5
            KK = 5
            for cl in this_round_clients:

                optimizer = make_optimizer(models[cl].prompt_learner,models[cl].cquery,base_lr=args.learning_rate)#.AdamW

                for e in range(1):
                    for i, (images, label,tar_idx) in enumerate(tqdm(client_dataloaders[cl])):
                        alpha = 0.5
                        optimizer.zero_grad()
                        with torch.no_grad():
                            x, x_q = (
                                visualmodel(images[0].to("cuda")),
                                visualmodel(images[1].to("cuda")),
                            )

                        domainout,softmax_out1 = models[cl](x_q,cls_only=True)                      
                        with torch.no_grad():
                            if gf==0:
                                p = tempmodel(x,True)
                            else:
                                p = tempmodel[cl](x,True)
                            output_t = nn.Softmax(-1)(p)
                            b = output_t.max(1)[0]
                            pseudo_labeled = output_t.max(1)[1]

                        qzhi = 0.75
                        loss = F.cross_entropy(softmax_out1[torch.where(b>qzhi)], pseudo_labeled[torch.where(b>qzhi)])
                        if args.alpha<0:
                            loss += F.cross_entropy(domainout, label.cuda()*0+cl)
                        else:
                            loss += F.cross_entropy(domainout, label.cuda()*0+cl//5)

                        loss.backward()
                        optimizer.step()



            weights = [1/len(this_round_clients)]*len(this_round_clients)
            update_query = []

            prompt = [] 
            local_state  = models[0].prompt_learner.state_dict()
            for k,cl in enumerate(this_round_clients):
                client_state = models[cl].prompt_learner.state_dict()
                for st in local_state:
                    if k==0:
                        local_state[st] = client_state[st]*weights[k]
                    else:
                        local_state[st] += client_state[st]*weights[k]

                update_query.append(models[cl].state_dict()['cquery.weight'].unsqueeze(0))

            global_model.prompt_learner.load_state_dict(local_state,strict=False)
            global_model.load_state_dict({'cquery.weight': torch.mean(torch.cat(update_query,dim=0),dim=0),},strict=False)




    if args.alpha<0.1:
        allfine = 4
    else:
        allfine = 20
    for fe in range(allfine):
        

        for m,client in enumerate(models):
            client.prompt_learner.load_state_dict(global_model.prompt_learner.state_dict(),strict=False)
            client.cquery.load_state_dict(global_model.cquery.state_dict(),strict=False)
            
        if fe == 0:
            fea_data,score_data,cls_data,cls_sim_data= [],[],[],[]
            for cl,client in enumerate(tqdm(models)):
                fea_bank = torch.randn(lens[cl], 512).half()
                score_bank = torch.randn(lens[cl], 345).cuda().half()
                cls_sim_bank = torch.randn(lens[cl], 345).cuda().half()
                for i, (images, label,indx) in enumerate((client_dataloaders[cl])):

                    with torch.no_grad():
                        image = images[0].cuda()
                        output_norm = visualmodel(image)
                        _,outputs1= client(output_norm,cls_only=True)
                        output_norm = output_norm / output_norm.norm(dim=-1, keepdim=True)
                        outputs = nn.Softmax(-1)(outputs1)
                    cls_sim_bank[indx] = outputs1.detach().clone()
                    fea_bank[indx] = output_norm.detach().clone().cpu()
                    score_bank[indx] = outputs.detach().clone()  # .cpu()
                
                temp = torch.randn(345, 345).cuda().half()
                pl = cls_sim_bank.max(1)[1]
                for i in range(345):
                    temp[i] = torch.mean(cls_sim_bank[torch.where(pl==i)],0)
                cls_data.append(temp)
                cls_sim_data.append(cls_sim_bank)
                fea_data.append(fea_bank)
                score_data.append(score_bank)
                
            init_cls = sum(cls_data)/len(cls_data)


        logger.info(f'------------- federated {fe}-th  --------------------------')
        print('------------- federated ',fe,'-th  --------------------------')
        if args.alpha<0:
            this_round_clients = list(range(6))
        else:
            this_round_clients = [np.random.choice(list(range(5*k,5*k+5)), 1, replace=False)[0] for k in range(len(domains))]
                
        alpha0 = 1
        beta0 = 5
        KK = 5
        
        for cl in this_round_clients:

            optimizer = make_optimizer(models[cl].prompt_learner,models[cl].cquery,base_lr=args.learning_rate)#.AdamW

            for e in range(1):
                # mmm=0
                for i, (images, label,tar_idx) in enumerate(tqdm(client_dataloaders[cl])):
                    alpha = 0.5
                    optimizer.zero_grad()
                    with torch.no_grad():
                        x, x_q = (
                            visualmodel(images[0].to("cuda")),
                            visualmodel(images[1].to("cuda")),
                        )

                    domainout,logits = models[cl](x,cls_only=True) 

                    
                    softmax_out = nn.Softmax(-1)(logits)
                    _,out_q1 = models[cl](x_q,cls_only=True) 
                    out_q = nn.Softmax(-1)(out_q1)
                    loss = torch.mean(torch.sum(-1 * torch.log_softmax(out_q,1) * score_data[cl][tar_idx], dim=1))
                    if args.alpha<0:
                        loss += F.cross_entropy(domainout, label.cuda()*0+cl)
                    else:
                        loss += F.cross_entropy(domainout, label.cuda()*0+cl//5)


                    with torch.no_grad():
                        output_f_norm = x / x.norm(dim=-1, keepdim=True)
                        output_f_ = output_f_norm.cpu().detach().clone()

                        fea_data[cl][tar_idx] = output_f_.detach().clone().cpu()
                        score_data[cl][tar_idx] = softmax_out.detach().clone()
                        cls_sim_data[cl][tar_idx] = logits.detach().clone()
                        
                        distance = (output_f_.cuda() @ fea_data[cl].T.cuda()).cpu()

                        _, idx_near = torch.topk(distance.float(), dim=-1, largest=True, k=KK + 1)
                        idx_near = idx_near[:, 1:]  # batch x K
                        score_near = score_data[cl][idx_near]  # batch x K x C


                    softmax_out_un = softmax_out.unsqueeze(1).expand(-1, KK, -1)  # batch x K x C

                    loss += torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)) # Equal to dot product
                    # print(loss.item())
                    mask = torch.ones((x.shape[0], x.shape[0]))
                    diag_num = torch.diag(mask)
                    mask_diag = torch.diag_embed(diag_num)
                    mask = mask - mask_diag
                    scopy = softmax_out.T  # .detach().clone()#

                    dot_neg = softmax_out @ scopy  # batch x batch

                    dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
                    neg_pred = torch.mean(dot_neg)
                    loss += neg_pred * alpha
                    

                    loss.backward()
                    optimizer.step()



        weights = [1/len(this_round_clients)]*len(this_round_clients)
        update_query = []

        prompt = [] 
        local_state  = models[0].prompt_learner.state_dict()
        for k,cl in enumerate(this_round_clients):
            client_state = models[cl].prompt_learner.state_dict()
            for st in local_state:
                if k==0:
                    local_state[st] = client_state[st]*weights[k]
                else:
                    local_state[st] += client_state[st]*weights[k]

            update_query.append(models[cl].state_dict()['cquery.weight'].unsqueeze(0))

        global_model.prompt_learner.load_state_dict(local_state,strict=False)
        global_model.load_state_dict({'cquery.weight': torch.mean(torch.cat(update_query,dim=0),dim=0),},strict=False)



        if fe in [4,10,15]:
            torch.save(global_model.prompt_learner, f'./pretrain_prompt{args.alpha}_de{fe}.pth')
            torch.save(global_model.cquery, f'./cquery{args.alpha}_de{fe}.pth')
        if fe==allfine-1:

            logger.info('epoch: %s' % str(fe))
            for te,test_loader in enumerate(client_testloaders):
                top1,topk = evala(global_model,visualmodel,test_loader)
                logger.info('top1: %s' % str(top1))
                print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)  


