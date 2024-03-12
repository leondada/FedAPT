# FedAPT
This repository is the implementation of "Federated Adaptive Prompt Tuning for Multi-domain Collaborative Learning". 

The original code has not been organized yet, and we plan to refactor the code in the future to improve readability.
## Requirements
### Dependencies
```
Python 3.7.11
torch 1.10.2
torchvision 0.12.0+cu113
tqdm 4.63.0
numpy
clip 1.0
```
### Datasets
A `datapath` should be defined (such as `datapath='/home/share/DomainNet/'`). The directory structure should be
```
/home/share/DomainNet/
│       
└───clipart
│   │...
└───infograph
│   │...
...
└───sketch
│   │...   
```
Download and unzip the [DomainNet](http://ai.bu.edu/M3SDA/) dataset to datapath.

## Training

Each domain has one client:
```
PromptFL: python prompt-promptfl.py --logname 'xxxx' --datapath 'xxxx' --data 'domainnet'
FedAPT (ours): python prompt-ours.py --logname 'xxxx' --datapath 'xxxx' --data 'domainnet'
```

Each domain has five client:
```
PromptFL: python prompt-promptfl-device.py --logname 'xxxx' --datapath 'xxxx' --data 'domainnet' --alpha 0.5
FedAPT (ours): python prompt-ours-device.py --logname 'xxxx' --datapath 'xxxx' --data 'domainnet' --alpha 0.5
```


