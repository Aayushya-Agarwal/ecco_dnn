import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import matplotlib.pyplot as plt
import datetime
import random
import argparse

if __name__ == "__main__":
    # available optim_: "EC/", "SGD/", "Adam/", "AdaGrad/", "RMSprop/"
    parser = argparse.ArgumentParser(description='Random Search for PyTorch CIFAR10 Training')
    parser.add_argument('--optim_', default="EC", type=str, help='optimizer method: EC SGD Adam AdaGrad RMSProp')

    parser.add_argument('--model', default='resnet18', type = str, help='Model')
    parser.add_argument('--num_search', default=100, type=int, help='Number of Random Searches')

    args = parser.parse_args()
    num_iter = args.num_search
    model_str = args.model
    optim_ = args.optim_ + '/'
    
    if model_str == "resnet18_update":
        limiting_flag = 1
    else:
        limiting_flag = 0

    if os.path.isdir("csv_files/") == 0:
        os.mkdir("csv_files/")
        
    # Generating cases for Adam optimizer
    if optim_ == 'Adam/':
        csv_path = "csv_files/adam_result.csv"
        for _ in range(num_iter):
            seed_ = random.randint(1,100)
            lr = round(random.uniform(0,1), 6)
            t_max = random.randint(1,300)
            beta_1 = round(random.uniform(0,1), 6)
            beta_2 = round(random.uniform(0,1), 6)
            os.system(f"python train_cifar100.py --optim_ {optim_} --lr {lr} --beta_1 {beta_1} --beta_2 {beta_2} --t_max {t_max} --seed {seed_} --model_str {model_str} --csv_path {csv_path}")

    # Generating cases for SGD optimizer
    elif optim_ == 'SGD/':
        csv_path = "csv_files/sgd_result.csv"
        for _ in range(num_iter):
            seed_ = random.randint(1,100)
            lr = round(random.uniform(0,1), 6)
            wd = 10 ** random.randint(-8, -1)
            t_max = random.randint(50,500)
            os.system(f"python train_cifar100.py --optim_ {optim_}  --lr {lr} --wd {wd} --t_max {t_max} --seed {seed_} --model_str {model_str} --csv_path {csv_path}")

     
    # Generating cases for AdaGrad optimizer
    elif optim_=='AdaGrad/':
        csv_path = "csv_files/adagrad_result.csv"
        for _ in range(num_iter):
            seed_ = random.randint(1,100)
            lr = round(random.uniform(0,1), 6)
            wd = 10 ** random.randint(-8, -1)
            ld = round(random.uniform(0,1), 6)
            os.system(f"python train_cifar100.py --optim_ {optim_}  --lr {lr} --wd {wd} --ld {ld} --seed {seed_} --model_str {model_str} --csv_path {csv_path}")

    # Generating cases for RMSprop optimizer
    elif optim_=='RMSprop/':
        csv_path = "csv_files/rmsprop_result.csv"
        for _ in range(num_iter):
            seed_ = random.randint(1,100)
            lr = round(random.uniform(0,1), 6)
            alpha = round(random.uniform(0,1), 6)
            wd = 10 ** random.randint(-8, -1)
            os.system(f"python train_cifar100.py --optim_ {optim_}  --lr {lr} --alpha {alpha} --wd {wd} --seed {seed_} --model_str {model_str} --csv_path {csv_path}")
   
    # Generating cases for EC optimizer
    elif optim_ =='EC/':
        csv_path = "csv_files/ec_result.csv"
        for _ in range(num_iter):
            alpha_0 = round(random.uniform(0,1), 6)
            eta = round(random.uniform(0,100), 6)
            seed_ = random.randint(1,100)
            os.system(f"python train_cifar100.py --optim_ {optim_} --alpha_0 {alpha_0}  --eta {eta} --seed {seed_} --model_str {model_str} --csv_path {csv_path} --limiting_flag {limiting_flag}")

