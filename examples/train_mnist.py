'''Train MNIST with PyTorch.'''
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

import csv
import ipdb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--optim_', default="Adam/", type=str, help='optimizer method')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--alpha_0', default=0.01, type=float, help='alpha_0 for ecco_dnn')
parser.add_argument('--alpha', default=0.1, type=float, help='alpha for RMSprop')
parser.add_argument('--wd', default = 0.0, type = float, help = 'weight decay')
parser.add_argument('--ld', default = 0.0, type = float, help = 'lr decay')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--beta_1', default=0.9, type=float, help='Adam beta_1')
parser.add_argument('--beta_2', default=0.999, type=float, help='Adam beta_2')
parser.add_argument('--t_max', default=200, type=int, help='Tmax for schedular')
parser.add_argument('--eta', default=0.3, type=float, help='eta')
parser.add_argument('--seed', default=66, type=int, help='Random seed')
parser.add_argument('--epoch', default=10, type=int, help='epoch number')
parser.add_argument('--csv_path', default="result.csv", type=str, help='result csv path')
parser.add_argument('--model_str', default="SimpleDNN", type=str, help='used model')
parser.add_argument('--limiting_flag', default=0, type=int, help='whether we have limiting step')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 1
# # results will be different if miss any following two lines
seed_ = args.seed
torch.manual_seed(seed_)
torch.backends.cudnn.deterministic = True

num_epoch = args.epoch
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
limiting_flag = args.limiting_flag

# Data
print('==> Preparing data..')
bach_size=100

trainset=torchvision.datasets.MNIST('content',train=True,transform=transforms.ToTensor(),download=True)
testset=torchvision.datasets.MNIST('content',train=False,     
              transform=transforms.ToTensor(),download=True)
trainloader=torch.utils.data.DataLoader(dataset=trainset, 
                                  batch_size=bach_size,shuffle=True)
testloader=torch.utils.data.DataLoader(dataset=testset, 
                                  batch_size=bach_size,shuffle=True)


# Model
print('==> Building model..')
net_str = args.model_str

# net_str = "resnet18"
# net_str = "lenet5"
# net_str = "senet18"

input_size=784   #28X28 pixel of image
hidden_size1=200 #size of 1st hidden layer(number of perceptron)
hidden_size2=150 #size of second hidden layer
hidden_size3=100 #size of third hidden layer
hidden_size=80   #size of fourth hidden layer
output =10       #output layer

if net_str == "SimpleDNN":
    net = SimpleDNN(input_size,hidden_size1,hidden_size2
                       ,hidden_size3,hidden_size,output)
if net_str == "SimpleDNNUpdate":
    net = SimpleDNNUpdate(input_size,hidden_size1,hidden_size2
                       ,hidden_size3,hidden_size,output)
    

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

criterion = nn.CrossEntropyLoss()
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
    if limiting_flag:
        optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0 = args.alpha_0, limiting_layer_idx=[0,2,4,6], num_layers=10, layer_type="linear")
    else:
        optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0 = args.alpha_0)

start_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
if os.path.isdir("checkpoint/") == 0:
    os.mkdir("checkpoint/")
if os.path.isdir("checkpoint/mnist/") == 0:
    os.mkdir("checkpoint/mnist/")
parent_folder = "checkpoint/mnist/" + optimizer_method
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
    temp_list = []
    x_temp_list = []
    x_prime_list = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.reshape(-1,28*28)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if (net_str == "lenet5/"):
            outputs = net(inputs)[1]
        else:
            outputs = net(inputs)
            if limiting_flag:
                relu_input_list = net.ec_step_input # for limiting cases
        loss = criterion(outputs, targets)
        loss.backward()
        if limiting_flag:
            optimizer.step(relu_input_list) # for limiting cases
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / len(trainloader), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.reshape(-1,28*28)
            inputs, targets = inputs.to(device), targets.to(device)
            if (net_str == "lenet5/"):
                outputs = net(inputs)[1]
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # torch.save(state, checkpoint_path + "/ckpt.pth")
        best_acc = acc
    return test_loss / len(testloader), 100.*correct/total

train_loss_list = []
test_loss_list = []

for epoch in range(start_epoch, start_epoch+num_epoch):
    log_file.write("Epoch: " +str(epoch) + "\n")
    train_batch_loss, train_acc = train(epoch)
    final_train_acc = train_acc
    log_file.write("train loss: " + str(train_batch_loss) + " | train acc: " + str(train_acc) + "\n")
    test_batch_loss, test_acc = test(epoch)
    log_file.write("test loss: " + str(test_batch_loss) + " | test acc: " + str(test_acc) + "\n")
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
