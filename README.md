# ECCO-DNN Optimization Algorithm

This repository contains the implementation of the ECCO-DNN optimization algorithm in PyTorch. The code needed to reproduce the experiments are available in the examples/ folder.

Due to randomness from the data generation and initialization, the results may differ from the results in the paper, however the conclusions remain the same. 


## Prerequisites
- Python 3.6+
- PyTorch 1.0+


## Running Examples
CIFAR10 Example for Random Search: python examples/random_search_cifar10.py 

LSTM for Power System Dataset Example: python examples/random_search_lstm.py 
The LSTM model is trained on a power systems dataset provided by [1]

MNIST Example: python examples/random_search_mnist.py 

CIFAR100 Example for Random Search: python examples/random_search_cifar10.py 

## Installing ECCO-DNN
ECCO-DNN can be installed by
1. Add ecco_dnn.py to the model folder
2. Import ECCO-DNN into the main file by adding: 
`from ecco_dnn import ECCO_DNN`

## Initializing ECCO-DNN Optimizer
The ECCO-DNN optimizer can be initialized with two parameters: a maximum LTE tolerance denoted as eta, and an initial step size denoted as alpha_0.
For example:
```
optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0)
```

If the user can explicitly define the layers that are part of the activation limiting (refer to [XX]), then the index of layers that are limited , and the type of activation layer (e.g., batch-norm) should be passed as a list to the optimizer. For example, the layers for the 'resnet18_update' model is provided as:

```
limiting_layer_idx = [1, 4, 10, 16, 25, 31, 40, 46, 55]
optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0, limiting_layer_idx=limiting_layer_idx, num_layers = 62, layer_type="bn")
```


The ECCO-DNN optimizer can then be called by:
```
limiting_layer_idx = [1, 4, 10, 16, 25, 31, 40, 46, 55]
optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0, limiting_layer_idx=limiting_layer_idx, num_layers = 62, layer_type="bn")
```


## Citing ECCO-DNN

[1] George Hebrail and Alice Berard. Individual household electric power consumption data set. Aug 2012
