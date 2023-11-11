'''Train LSTM with PyTorch.'''

#based on https://github.com/chandimap/LSTM-To-Predict-Household-Electric-Power-Consumption
# data from http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_dir)
from kapperkermit.models import *
from kapperkermit.ecco_dnn import ECCO_DNN
from kapperkermit.utils import progress_bar
from kapperkermit.utils import write_to_csv

import matplotlib.pyplot as plt
import datetime

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import csv
import ipdb
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#parser.add_argument('--optim_', default="EC/", type=str, help='optimizer method')
parser.add_argument('--optim_', default="Adam/", type=str, help='optimizer method')
parser.add_argument('--lr', default=0.000726, type=float, help='learning rate')
parser.add_argument('--alpha_0', default=0.01, type=float, help='alpha_0 for ecco_dnn')
parser.add_argument('--alpha', default=0.301401, type=float, help='alpha for RMSprop')
parser.add_argument('--wd', default = 1e-5, type = float, help = 'weight decay')
parser.add_argument('--ld', default = 0.060346, type = float, help = 'lr decay')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--beta_1', default=0.841, type=float, help='Adam beta_1')
parser.add_argument('--beta_2', default=0.853, type=float, help='Adam beta_2')
parser.add_argument('--t_max', default=226, type=int, help='Tmax for schedular')
parser.add_argument('--eta', default=0.3, type=float, help='eta')
parser.add_argument('--seed', default=66, type=int, help='Random seed')
parser.add_argument('--epoch', default=50, type=int, help='epoch number')
parser.add_argument('--csv_path', default="result.csv", type=str, help='result csv path')
parser.add_argument('--model_str', default="lstm", type=str, help='used model')
args = parser.parse_args()


class PowerDataset(Dataset):

    def __init__(self, x_data, y_data):

        super(Dataset,self).__init__()

        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return X_train.shape[0] 


    def __getitem__(self, idx):

        #if isinstance(idx, Tensor):
        #if torch.istensor(idx):
        #idx = idx.tolist()

        #print("IDX: ",idx)
        #print("XDATA: ",self.x_data[idx,:,:])
        #print("shape x_data: ",np.shape(self.x_data))
        #print("shape y_data: ",np.shape(self.y_data))
        return self.x_data[idx,:,:], self.y_data[idx]


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 0
# # results will be different if miss any following two lines
seed_ = args.seed
torch.manual_seed(seed_)
torch.backends.cudnn.deterministic = True

num_epoch = args.epoch
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

power = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'Date_Time' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='Date_Time')
droping_list_all=[]
for j in range(0,7):
    if not power.iloc[:, j].notnull().all():
        droping_list_all.append(j)   
power.drop(['Voltage'],1,inplace=True)

def data_prep(data, n_in=1, n_out=1, dropnan=True):
    names, cols = list(), list()
    data_frame = pd.DataFrame(data)
    n = 1 if type(data) is list else data.shape[1]
    
    for x in range(n_in, 0, -1):  #  Input Sequence (t-n, ... t-1)
        cols.append(data_frame.shift(x))
        names += [('var%d(t-%d)' % (y+1, x)) for y in range(n)]
    
    for x in range(0, n_out):  #  Forecast Sequence (t, t+1, ... t+n)
        cols.append(data_frame.shift(-x))
        if x == 0:
            names += [('var%d(t)' % (y+1)) for y in range(n)]
        else:
            names += [('var%d(t+%d)' % (y+1, x)) for y in range(n)]
    
    z = pd.concat(cols, axis=1)  #  Putting It All Together
    z.columns = names
    
    if dropnan:  #  Dropping Rows With NaN Values
        z.dropna(inplace=True)
    return z

over_hour = power.resample('h').mean() 
over_hour.shape

results = over_hour.values 
#  Normalizing Features
scaler = MinMaxScaler(feature_range=(0, 1))
s = scaler.fit_transform(results)
s.shape

#  Framing As Supervised Learning
r = data_prep(s, 1, 1)
#  Dropping Columns which are not predicted
r.drop(r.columns[[7,8,9,10,11]], axis=1, inplace=True)
print(r.head())

#  Splitting Into Train & Test Sets
results = r.values
results = torch.from_numpy(results).float()

duration = 365 * 24
train = results[:duration, :]
test = results[duration:, :]


#  Splitting Into Input & Outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]


# Rehaping The Input Into The 3D Format As Expected By LSTMs, Namely [samples, timesteps, features].
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  #  Reshaping Input To Be 3D [samples, timesteps, features]
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


train_dataset = PowerDataset(X_train, y_train)

test_dataset = PowerDataset(X_test, y_test)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2)


# Model
net_str = args.model_str

print('==> Building model..')
if net_str == "lstm":
    net = LSTM()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer_method = args.optim_  

if (optimizer_method == "SGD/"):
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)


if (optimizer_method == "Adam/"):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas = (args.beta_1, args.beta_2))                    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

if (optimizer_method == "AdaGrad/"):
    optimizer = torch.optim.Adagrad(net.parameters(), lr = args.lr, lr_decay = args.ld, weight_decay = args.wd)

if (optimizer_method == "RMSprop/"):
    optimizer = torch.optim.RMSprop(net.parameters(), lr = args.lr, alpha = args.alpha, weight_decay = args.wd)

if (optimizer_method == "EC/"):
    optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0)
    


start_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
if os.path.isdir("checkpoint/") == 0:
    os.mkdir("checkpoint/")
if os.path.isdir("checkpoint/power_data/") == 0:
    os.mkdir("checkpoint/power_data/")
parent_folder = "checkpoint/power_data/" + optimizer_method
if os.path.isdir(parent_folder) == 0:
    os.mkdir(parent_folder)
checkpoint_path = parent_folder + net_str + "_" +  start_time
# print(checkpoint_path)
os.mkdir(checkpoint_path)
log_file = open(checkpoint_path + "/log.txt", "w")

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        #targets = targets.to(device)
        optimizer.zero_grad()
        #ipdb.set_trace()
        outputs = net(inputs)
        #outputs = outputs.type(torch.LongTensor)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #_, predicted = outputs.max(1)
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'MSE Loss: %.6f'
                     % (train_loss/(batch_idx+1)))
    return train_loss/len(trainloader)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f '
                         % (test_loss/(batch_idx+1)))

    # Save checkpoint.
    '''acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # torch.save(state, checkpoint_path + "/ckpt.pth")
        best_acc = acc'''
    
    return test_loss / len(testloader)

train_loss_list = []
test_loss_list = []
for epoch in range(start_epoch, start_epoch+num_epoch):
    log_file.write("Epoch: " +str(epoch) + "\n")
    train_batch_loss = train(epoch)
    # ipdb.set_trace()
    final_train_acc = train_batch_loss #train_acc
    log_file.write("train loss: " + str(train_batch_loss) + " | train acc: " + str(train_batch_loss) + "\n")
    test_batch_loss = test(epoch)
    log_file.write("test loss: " + str(test_batch_loss) + " | test acc: " + str(test_batch_loss) + "\n")
    log_file.flush()
    train_loss_list.append(train_batch_loss)
    test_loss_list.append(test_batch_loss)
    if optimizer_method == "Adam/" or optimizer_method == "SGD/":
        scheduler.step()

# store the result in corresponding csv file
write_to_csv(args, final_train_loss = train_loss_list[-1], final_train_acc = final_train_acc, 
                checkpoint_path = checkpoint_path)

plt.figure()
plt.plot(range(start_epoch, start_epoch+num_epoch), train_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.savefig(checkpoint_path + "/train_loss.png")

plt.figure()
plt.plot(range(start_epoch, start_epoch+num_epoch), test_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.savefig(checkpoint_path + "/test_loss.png")

train_loss_list = torch.tensor(train_loss_list)
torch.save(train_loss_list, checkpoint_path + "/train_loss.pt")
test_loss_list = torch.tensor(test_loss_list)
torch.save(test_loss_list, checkpoint_path + "/test_loss.pt")
