import torch
from  torch.optim import Optimizer
import numpy as np
import time 
import ipdb

class ECCO_DNN(Optimizer):
    def __init__(self, params, eta = 0.3, alpha_0=0.01, limiting_layer_idx = [], num_layers = 0, layer_type = ""):

        #Local Truncation Error upper limit
        self.eta = eta 

        #binary vector describing layers that will be limited using activation function limiting
        self.limiting_layer_mask = torch.zeros(num_layers) 
        self.limiting_layer_mask[limiting_layer_idx] = 1
        self.layer_type = layer_type
        defaults = dict(prev_theta = [], prev_gk = [], prev_delta_theta = [], prev_flag = False, prev_alpha = alpha_0)
        super(ECCO_DNN, self).__init__(params, defaults)
    
    #function calculates the trajectory shaping matrix inverse(Z) 
    def calcZetaInv(self, gk, prev_gk, alpha_k):

        # difference in the gradient
        g_diff = torch.sub(gk, prev_gk)
        alpha_k_inv = 1 / alpha_k

        #approximation to time-derivative of gradient
        dgdiff_dt = torch.mul(g_diff, alpha_k_inv)

        zeta_inv_sq = gk * dgdiff_dt

        #limiting the values of inverse(Z^2)<=1
        zeta_inv_sq[zeta_inv_sq<=1] = 1


        zeta_inv = torch.sqrt(zeta_inv_sq)

        return zeta_inv


    #function calculates the step-size to limit the Local truncation error (delta_t_tau)
    def calcDeltat(self, delta_theta, prev_delta_theta, eta):
        delta_t_tau = min(2 * eta / torch.max(torch.abs(prev_delta_theta - delta_theta)), 0.5)
        return delta_t_tau
    
    #function calculates the step size to limit activation layer from switching
    def calcActivationLimitingDelta(self, input_idx, theta, next_layer_theta, delta_weight, delta_bias):

        #the input to the activation function is a batch-norm layer
        if (self.layer_type == "bn"):
            bn_layer = torch.nn.BatchNorm2d(self.relu_input_list[input_idx].shape[1])
            weight_ = torch.nn.parameter.Parameter(theta.clone())
            bias_ = torch.nn.parameter.Parameter(next_layer_theta.clone())
            bn_layer.weight = weight_
            bn_layer.bias = bias_

            theta_a = bn_layer(self.relu_input_list[input_idx])

            delta_bn_layer = torch.nn.BatchNorm2d(self.relu_input_list[input_idx].shape[1])
            delta_weight = torch.nn.parameter.Parameter(delta_weight)
            delta_bias = torch.nn.parameter.Parameter(delta_bias)
            delta_bn_layer.weight = delta_weight
            delta_bn_layer.bias = delta_bias
            delta_theta_a = delta_bn_layer(self.relu_input_list[input_idx])
        
        #the input to the activation function is a linear layer
        elif (self.layer_type == "linear"):
            linear_layer = torch.nn.Linear(theta.data.shape[1], theta.data.shape[0])
            weight_ = torch.nn.parameter.Parameter(theta.clone())
            bias_ = torch.nn.parameter.Parameter(next_layer_theta.clone())
            linear_layer.weight = weight_
            linear_layer.bias = bias_
            theta_a = linear_layer(self.relu_input_list[input_idx])

            delta_linear_layer = torch.nn.Linear(theta.data.shape[1], theta.data.shape[0])
            delta_weight = torch.nn.parameter.Parameter(delta_weight)
            delta_bias = torch.nn.parameter.Parameter(delta_bias)
            delta_linear_layer.weight = delta_weight
            delta_linear_layer.bias = delta_bias
            delta_theta_a = delta_linear_layer(self.relu_input_list[input_idx])
        
        #calculate the step size to limit the input to the activation layer from switching past a value of 0
        delta_i = min(1 / torch.abs(torch.min(delta_theta_a / theta_a)), 1)

        return delta_i



    def step(self, relu_input_list = []):
        self.relu_input_list = relu_input_list
        for param_group in self.param_groups:
            thetas = param_group['params']
            prev_theta = []
            prev_gk = []
            prev_delta_theta = []

            input_idx = 0
            skip_flag = False
            for i in range(len(thetas)):
                theta = thetas[i]

                #If this is the first iteration
                if (param_group['prev_flag'] == False):

                    #append parameter values to vector of parameter values from previous iteration
                    prev_theta.append(theta.data.detach().clone())
                    #append gradient values to vector of gradient values from previous iteration
                    prev_gk.append(theta.grad.detach().clone())

                    zeta_inv = 1

                    #step size is initial step size
                    alpha_k = param_group['prev_alpha']
                    prev_delta_theta.append(theta.grad.detach().clone())

                    #update parameter values
                    theta.data = theta.data - alpha_k * zeta_inv * theta.grad
                    param_group['prev_alpha'] = alpha_k
                else:
                    # skip bias layer for activation input
                    if (skip_flag == True):
                        skip_flag = False
                        continue

                    #append parameter values to vector of parameter values from previous iteration
                    prev_theta.append(theta.data.detach().clone())
                    #append gradient values to vector of gradient values from previous iteration
                    prev_gk.append(theta.grad.detach().clone())

                    #calculate trajectory shaping matrix, inv(Z)
                    zeta_inv = self.calcZetaInv(theta.grad.detach().clone(), param_group['prev_gk'][i],
                                     param_group['prev_alpha'])
                    
                    #change in parameters 
                    delta_theta = zeta_inv * theta.grad.detach().clone()

                    #calculate step size to bound Local Truncation Error by value, eta
                    delta_t_tau = self.calcDeltat(delta_theta, param_group['prev_delta_theta'][i], self.eta)
                    prev_delta_theta.append(delta_theta.detach().clone())


                    # If optimizing for an activation layer
                    if (len(self.limiting_layer_mask) != 0 and self.limiting_layer_mask[i] == 1 and self.layer_type != ""):
                        # compute delta_weight
                        delta_weight  = - delta_t_tau * zeta_inv * theta.grad
                        param_group['prev_alpha'] = delta_t_tau

                        # compute delta_bias of the next layer
                        next_layer_theta = thetas[i+1]
                        prev_theta.append(next_layer_theta.data.detach().clone())
                        prev_gk.append(next_layer_theta.grad.detach().clone())
                        next_zeta_inv = self.calcZetaInv(next_layer_theta.grad.detach().clone(), param_group['prev_gk'][i+1],
                                     param_group['prev_alpha'])
                        next_layer_delta_theta = next_zeta_inv * next_layer_theta.grad.detach().clone()
                        next_layer_delta_t_tau = self.calcDeltat(next_layer_delta_theta, param_group['prev_delta_theta'][i+1], self.eta)
                        prev_delta_theta.append(next_layer_delta_theta.detach().clone())
                        delta_bias = - next_layer_delta_t_tau * next_zeta_inv * next_layer_theta.grad

                        #calculating step size for limiting activation layer
                        delta_i = self.calcActivationLimitingDelta(input_idx, theta, next_layer_theta, delta_weight, delta_bias)
                        # add lower boundary of 1e-6 prevent the value to round to 0
                        # calculate alpha_k for weight layer
                        alpha_k = max((delta_i * delta_t_tau), 1e-6)

                        #update step using new alpha_k
                        theta.data =  theta.data - alpha_k * delta_theta
                        # calculate alpha_k for bias layer
                        alpha_k = max(delta_i * next_layer_delta_t_tau, 1e-6)
                        next_layer_theta.data = next_layer_theta.data - alpha_k * next_layer_delta_theta
                        param_group['prev_alpha'] = alpha_k

                        skip_flag =  True
                        input_idx += 1
                    else:
                        alpha_k = delta_t_tau
                        theta.data = theta.data - alpha_k * delta_theta
                        param_group['prev_alpha'] = alpha_k

            if (param_group['prev_flag'] == False):
                param_group['prev_flag'] = True
            param_group['prev_theta'] = prev_theta
            param_group['prev_gk'] = prev_gk
            param_group['prev_delta_theta'] = prev_delta_theta


