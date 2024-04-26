import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms

import os 
import cv2
import copy 
import random 
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
from dataloader import PotteryDataset
from architecture import TwoLayerCNN, FourLayerCNN

parser = argparse.ArgumentParser(description='Archaeological classifier for test')
parser.add_argument('--batch_size',default=128, type=int, help='batch_size')
parser.add_argument('--seed',default=0, type=int, help='random seed ')
parser.add_argument('--gpu',default=5, type=str, help='which gpu to use ')
parser.add_argument('--datapath',default='./data_sample', type=str, help='data path')
parser.add_argument('--dataname',default='zizhu', type=str, help='data name')
parser.add_argument('--savepath',default='./results', type=str, help='result save path')
parser.add_argument('--arch',default='twolayercnn', type=str, help='archtecture ')
parser.add_argument('--num_cls',default=3, type=int, help='number of classes')
args = parser.parse_args()
#获得传入的参数
print(args)

# device 
device = torch.device('cuda:%s'%args.gpu)

if args.arch == 'twolayercnn':
    model = TwoLayerCNN(output = args.num_cls)
elif args.arch == 'fourlayercnn':
    model = FourLayerCNN(output = args.num_cls)
elif args.arch == 'resnet18':
    model = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
    numfit = model.fc.in_features
    model.fc = nn.Linear(numfit, args.num_cls)
model = model.to(device)

model.load_state_dict(torch.load(os.path.join(args.savepath, 'model.pth')))

testpath = os.path.join(args.datapath, args.dataname)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

testset = PotteryDataset(datapath=testpath, transform=transformer)
test_n = len(testset)
print('Test data size:', test_n)
testloader = torch.utils.data.DataLoader(dataset= testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

clses = ['q%d'%i for i in range(args.num_cls)]
logs = open(os.path.join(args.savepath, 'logs.txt'), 'a+')


cls_pred = list(0. for i in range(len(clses))) 
model.eval()
names = []
outputs_prob = []
outputs_label = []
for i, (samples_names, samples, targets) in enumerate(testloader):
    samples = samples.to(device)
    samples = samples.float()
    targets = targets.to(device)
    targets = targets.long()
    names += list(samples_names)
    with torch.no_grad():
        outputs = model(samples)
        _, predicts = torch.max(outputs, 1)
        outputs_prob.append(outputs.detach().cpu().numpy())
        outputs_label.append(predicts.detach().cpu().numpy())
        for k in range(len(targets)):
            predict = predicts[k]
            cls_pred[predict] += 1
        # c = (predicts == targets).squeeze()
        # for k in range(len(targets)):
        #     target = targets[k]
        #     cls_correct[target] += c[k].item()
        #     cls_total[target] += 1
# print(len(names))
# print(names[:10])
outputs_prob = np.concatenate(outputs_prob)
outputs_label = np.concatenate(outputs_label)
# print(outputs_prob.shape)
# print(outputs_label.shape)

if args.num_cls == 3:
    df = pd.DataFrame({'img_name':names, 
                        'prob of 1':outputs_prob[:,0],
                        'prob of 2':outputs_prob[:,1],
                        'prob of 3':outputs_prob[:,2],
                        'predict':outputs_label,
                        })
elif args.num_cls == 4:
    df = pd.DataFrame({'img_name':names, 
                        'prob of 1':outputs_prob[:,0],
                        'prob of 2':outputs_prob[:,1],
                        'prob of 3':outputs_prob[:,2],
                        'prob of 4':outputs_prob[:,2],
                        'predict':outputs_label,
                        })

df.to_csv(os.path.join(args.savepath, '%s_test.csv'%args.dataname))
logs.write('\n =========%s========= \n'%args.dataname)
for i in range(len(clses)):
    print('Ratio of %5s: %2d %%' %( clses[i], 100 * cls_pred[i] /test_n))
    logs.write('Ratio of %5s: %2d %% \n' %(clses[i],  100 * cls_pred[i] /test_n))
# test_acc = 100 * sum(cls_correct) / sum(cls_total)
# print('Test total accuracy: %2d %%' %(test_acc) )
# logs.write('Test total accuracy: %2d %% \n' %(test_acc))

logs.close()

