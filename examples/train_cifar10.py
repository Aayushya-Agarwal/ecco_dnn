'''Train CIFAR10 with PyTorch.'''
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
parser.add_argument('--optim_', default="EC/", type=str, help='optimizer method')
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
parser.add_argument('--epoch', default=20, type=int, help='epoch number')
parser.add_argument('--csv_path', default="result.csv", type=str, help='result csv path')
parser.add_argument('--model_str', default="resnet18", type=str, help='used model')
parser.add_argument('--limiting_flag', default=0, type=int, help='whether we have limiting step')
args = parser.parse_args()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 0
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
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# trainset = torchvision.datasets.CIFAR100(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR100(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
net_str = args.model_str

if net_str == "googlenet":
    net = GoogLeNet()
if net_str == "resnet18":
    net = ResNet18()
if net_str == "senet18":
    net = SENet18()
if net_str == "resnet18_update":
    net = ResNet18Update()
if net_str == "lenet5":
    net = LeNet5(num_classes = 10)
if net_str == "resnet50":
    net =  ResNet50()
if net_str == "resnet101":
    net =  ResNet101()

# net = SENet18()
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
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
    if limiting_flag == 1 and net_str == "resnet18_update":
        limiting_layer_idx = [1, 4, 10, 16, 25, 31, 40, 46, 55]
        optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0, limiting_layer_idx=limiting_layer_idx, num_layers = 62, layer_type="bn")
    else:
        optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0)

start_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
if os.path.isdir("checkpoint/") == 0:
    os.mkdir("checkpoint/")
if os.path.isdir("checkpoint/cifar10/") == 0:
    os.mkdir("checkpoint/cifar10/")
parent_folder = "checkpoint/cifar10/" + optimizer_method
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
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if (net_str == "lenet5"):
            outputs = net(inputs)[1]
        else:
            outputs = net(inputs)
        if limiting_flag == 1:
            relu_input_list = [t.to(device) for t in net.relu_input_list]
            # relu_input_list = net.relu_input_list # for limiting cases
        loss = criterion(outputs, targets)
        loss.backward()
        if limiting_flag == 1:
            optimizer.step(relu_input_list)
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
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs)
            # outputs = net(inputs)[1] # for lenet5
            if (net_str == "lenet5"):
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
    # ipdb.set_trace()
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
