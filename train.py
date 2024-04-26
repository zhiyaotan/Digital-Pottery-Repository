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
from torch.utils.data.sampler import WeightedRandomSampler
from dataloader import PotteryDataset
from architecture import TwoLayerCNN, FourLayerCNN

parser = argparse.ArgumentParser(description='Archaeological classifier')
parser.add_argument('--num_epoches',default=10, type=int, help='num_epoches')
parser.add_argument('--batch_size',default=128, type=int, help='batch_size')
parser.add_argument('--eval_epoch',default=2, type=int, help='eval_epoch')
parser.add_argument('--lr',default=1e-4, type=float, help='learning_rate')
parser.add_argument('--seed',default=0, type=int, help='random seed ')
parser.add_argument('--gpu',default=5, type=str, help='which gpu to use ')
parser.add_argument('--datapath',default='./data_sample', type=str, help='data path')
parser.add_argument('--savepath',default='./results', type=str, help='result save path')
parser.add_argument('--arch',default='twolayercnn', type=str, help='archtecture ')
parser.add_argument('--num_cls',default=3, type=int, help='number of classes')
parser.add_argument('--upsample', default='False', type=str, help='weather to upsample')
# parser.add_argument('--dataset',default='changgui', type=str, help='which dataset to use')
parser.add_argument('--pretrained', default='True', type=str, help='pretrained bool type')
args = parser.parse_args()
#获得传入的参数
print(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
# device 
device = torch.device('cuda:%s'%args.gpu)


if args.arch == 'twolayercnn':
    model = TwoLayerCNN(output = args.num_cls)
elif args.arch == 'fourlayercnn':
    model = FourLayerCNN(output = args.num_cls)
elif args.arch == 'resnet18':
    if args.pretrained == 'True':
        model = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
    else:
        model = models.resnet18(pretrained = False)
    numfit = model.fc.in_features
    model.fc = nn.Linear(numfit, args.num_cls)
model = model.to(device)

if not os.path.exists(args.savepath):
    os.mkdir(args.savepath)
savepath = os.path.join(args.savepath, '%s_seed_%d' %(args.arch, args.seed))
# if args.pretrained == 'True' and args.arch =='resnet18':
#     savepath = savepath + '_pretrained'
if not os.path.exists(savepath):
    os.mkdir(savepath)

trainpath = os.path.join(args.datapath, 'train')
valpath = os.path.join(args.datapath, 'val')
testpath = os.path.join(args.datapath, 'test')

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = PotteryDataset(datapath=trainpath, transform=transformer)
valset = PotteryDataset(datapath=valpath, transform=transformer)
testset = PotteryDataset(datapath=testpath, transform=transformer)

print('Train data size:', len(trainset))
print('Val data size:', len(valset))
print('Test data size:', len(testset))

if args.upsample == 'True':
    if args.num_cls == 3:
        labels_counts = [18962, 12112, 4974]
    elif args.num_cls == 4:
        labels_counts = [18962, 12112, 4974, 24595]
    weights = [max(labels_counts) / labels_counts[i] for i in range(len(labels_counts))]
    weights = [weights[i] / sum(weights) for i in range(len(weights))]
    print(weights)
    sampler = WeightedRandomSampler(weights, len(weights) * max(labels_counts), replacement=True)
    trainloader = torch.utils.data.DataLoader(dataset= trainset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

else:
    trainloader = torch.utils.data.DataLoader(dataset= trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(dataset= valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(dataset= testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

train_n, val_n, test_n = len(trainloader.dataset), len(valloader.dataset), len(testloader.dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

clses = ['q%d'%i for i in range(args.num_cls)]

logs = open(os.path.join(savepath, 'logs.txt'), 'w')

best_acc = 0
for epoch in range(args.num_epoches):

    train_loss = 0.0 
    model.train()
    for i, (_, samples, targets) in enumerate(trainloader):
    
        samples = samples.to(device)
        samples = samples.float()
        # print(samples.shape)
        targets = targets.to(device)
        targets = targets.long()
        # targets = targets.reshape(targets.shape[0], 1)
        outputs = model(samples)
        # print(outputs.shape)
        # print(targets.shape)
        loss = criterion(outputs, targets)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * samples.shape[0]
        # break
    average_train_loss = train_loss / train_n
    print('Epoch : %d train loss : %.4f' %(epoch, average_train_loss))
    model.eval()
    if epoch % args.eval_epoch == 0:
        cls_correct = list(0. for i in range(len(clses)))
        cls_total = list(0. for i in range(len(clses))) 
        for i, (_, samples, targets) in enumerate(valloader):

            samples = samples.to(device)
            samples = samples.float()
            targets = targets.to(device)
            targets = targets.long()
            with torch.no_grad():
                outputs = model(samples)
                _, predicts = torch.max(outputs, 1)
                c = (predicts == targets).squeeze()
                for k in range(len(targets)):
                    target = targets[k]
                    cls_correct[target] += c[k].item()
                    cls_total[target] += 1
        for i in range(len(clses)):
            print('Epoch %d || Val accuracy of %5s: %2d %%' %(epoch, clses[i], 100 * cls_correct[i] /cls_total[i]))
            logs.write('Epoch %d || Val accuracy of %5s: %2d %% \n' %(epoch, clses[i], 100 * cls_correct[i] /cls_total[i]))
        val_acc = 100 * sum(cls_correct) / sum(cls_total)
        print('Val total accuracy: %2d %%' %(val_acc) )
        logs.write('Val total accuracy: %2d %% \n' %(val_acc))
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(savepath, 'model.pth'))

# figsavepath = os.path.join(savepath, 'figs')
# if not os.path.exists(figsavepath):
#     os.mkdir(figsavepath)

model.load_state_dict(best_model)
# mean = np.array([0.485, 0.456, 0.406])
# mean = mean.reshape(1,1,3)
# std = np.array([0.229, 0.224, 0.225])
# std = std.reshape(1,1,3)
cls_correct = list(0. for i in range(len(clses)))
cls_total = list(0. for i in range(len(clses))) 
model.eval()
for i, (_, samples, targets) in enumerate(testloader):
    samples = samples.to(device)
    samples = samples.float()
    targets = targets.to(device)
    targets = targets.long()
    with torch.no_grad():
        outputs = model(samples)
        _, predicts = torch.max(outputs, 1)
        c = (predicts == targets).squeeze()
        for k in range(len(targets)):
            target = targets[k]
            cls_correct[target] += c[k].item()
            cls_total[target] += 1
            # if targets[k] != predicts[k]:
                # img = (samples[k].detach().cpu().numpy().transpose(1,2,0) * std + mean) * 255
                # cv2.imwrite(os.path.join(figsavepath, '%d%d:%d-%d.jpg'%(i, k, targets[k], predicts[k])), img)
logs.write('\n =========Test========= \n')
for i in range(len(clses)):
    print('Test accuracy of %5s: %2d %%' %( clses[i], 100 * cls_correct[i] /cls_total[i]))
    logs.write('Test accuracy of %5s: %2d %% \n' %(clses[i], 100 * cls_correct[i] /cls_total[i]))
test_acc = 100 * sum(cls_correct) / sum(cls_total)
print('Test total accuracy: %2d %%' %(test_acc) )
logs.write('Test total accuracy: %2d %% \n' %(test_acc))

logs.close()

        

                 
        
        

    





