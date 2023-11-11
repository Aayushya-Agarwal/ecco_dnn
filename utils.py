'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - write_to_csv: store result in csv
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import csv


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# Function to write data to a CSV file
def write_to_csv_Adam(seed, lr, beta_1, beta_2, t_max, final_train_loss, final_train_acc, checkpoint_path, csv_path, model = "resnet18", num_epoch = 20, optimizer = "Adam"):
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['model', 'optimizer', 'seed','num_epoch', 'learning_rate',
                          'beta_1', 'beta_2', 'Tmax','final train loss', 'final train acc', 'checkpoint_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model, optimizer, seed, num_epoch, lr, beta_1, beta_2, t_max, final_train_loss, final_train_acc, checkpoint_path])

# Function to write data to a CSV file
def write_to_csv_sgd(seed, lr, wd, t_max, final_train_loss, final_train_acc, checkpoint_path, csv_path, model = "resnet18", num_epoch = 20, optimizer = "SGD"):
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['model', 'optimizer', 'seed','num_epoch', 'learning_rate', 'weight decay', 'Tmax',
                         'final train loss', 'final train acc', 'checkpoint_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model, optimizer, seed, num_epoch, lr, wd, t_max, final_train_loss, final_train_acc, checkpoint_path])

# Function to write data to a CSV file
def write_to_csv_adagrad(seed, lr, wd, ld, final_train_loss, final_train_acc, checkpoint_path, csv_path, model = "resnet18", num_epoch = 20, optimizer = "AdaGrad"):
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['model', 'optimizer', 'seed','num_epoch', 'learning_rate', 'weight decay', 'lr decay',
                         'final train loss', 'final train acc', 'checkpoint_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model, optimizer, seed, num_epoch, lr, wd, ld, final_train_loss, final_train_acc, checkpoint_path])

# Function to write data to a CSV file
def write_to_csv_rmsprop(seed, lr, alpha, wd, final_train_loss, final_train_acc, checkpoint_path, csv_path, model = "resnet18", num_epoch = 20, optimizer = "RMSprop"):
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['model', 'optimizer', 'seed','num_epoch', 'learning_rate', 'alpha', 'weight decay', 
                         'final train loss', 'final train acc', 'checkpoint_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model, optimizer, seed, num_epoch, lr, alpha, wd, final_train_loss, final_train_acc, checkpoint_path])

# Function to write data to a CSV file
def write_to_csv_ec(seed, alpha_0, eta,final_train_loss, final_train_acc, checkpoint_path, csv_path, model = "resnet18", num_epoch = 20, optimizer = "EC"):
    # print(csv_path)
    # print(model)
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['model', 'optimizer', 'seed','num_epoch', 'alpha_0', 'eta',
                          'final train loss', 'final train acc', 'checkpoint_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model, optimizer, seed, num_epoch, alpha_0, eta, final_train_loss, final_train_acc, checkpoint_path])

def write_to_csv(args, final_train_loss, final_train_acc, checkpoint_path):
    optimizer_method = args.optim_
    # print(optimizer_method)
    if optimizer_method == "EC/":
        write_to_csv_ec(seed = args.seed, alpha_0 = args.alpha_0, eta = args.eta,
                final_train_loss = final_train_loss, final_train_acc = final_train_acc,
                checkpoint_path = checkpoint_path, csv_path = args.csv_path, 
                model = args.model_str, num_epoch = args.epoch, optimizer = optimizer_method[:len(optimizer_method)-1])

    if optimizer_method == "Adam/":
        write_to_csv_Adam(seed = args.seed, lr = args.lr, 
                beta_1 = args.beta_1, beta_2=args.beta_2, t_max = args.t_max,
                final_train_loss = final_train_loss, final_train_acc = final_train_acc,
                checkpoint_path = checkpoint_path, csv_path = args.csv_path, 
                model = args.model_str, num_epoch = args.epoch, optimizer = optimizer_method[:len(optimizer_method)-1])

    if optimizer_method == "SGD/":
        write_to_csv_sgd(seed = args.seed, lr = args.lr, wd = args.wd, t_max = args.t_max,
                final_train_loss = final_train_loss, final_train_acc = final_train_acc,
                checkpoint_path = checkpoint_path, csv_path = args.csv_path, 
                model = args.model_str, num_epoch = args.epoch, optimizer = optimizer_method[:len(optimizer_method)-1])

    if optimizer_method == "AdaGrad/":
        write_to_csv_adagrad(seed = args.seed, lr = args.lr, wd = args.wd, ld = args.ld,
                final_train_loss = final_train_loss, final_train_acc = final_train_acc,
                checkpoint_path = checkpoint_path, csv_path = args.csv_path, 
                model = args.model_str, num_epoch = args.epoch, optimizer = optimizer_method[:len(optimizer_method)-1])

    if optimizer_method == "RMSprop/":
        write_to_csv_rmsprop(seed = args.seed, lr = args.lr, alpha = args.alpha, wd = args.wd,
                final_train_loss = final_train_loss, final_train_acc = final_train_acc,
                checkpoint_path = checkpoint_path, csv_path = args.csv_path, 
                model = args.model_str, num_epoch = args.epoch, optimizer = optimizer_method[:len(optimizer_method)-1])