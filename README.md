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

<details>
    <summary>Enabling Activation Limiting</summary>

To utilize activation-specific limiting, ecco_dnn requires the following parameteres to explicitly define the layers that are part of the activation limiting step:

    1. The function type before the activation layer: batch-norm("bn") or linear("linear")

    2. The number of learnable parameters of the model (num_layers)
    
    3. The list of weight parameter indices before selected activation layer (limiting_layer_idx)
    
    4. Input values for layer before selected activation layers (model.relu_input_list), which should be recorded in the forward function of the model.


In our examples (examples/train_cifar10.py), we apply the activation-limiting step to 9 activation layers of a Resnet18 architecture. The following figure highlights the specific layers of Resnet18 that are limited, as provided by `limiting_layer_idx`. 
During the forward propogation of this model, the input values for each layer in `limiting_layer_idx` are recorded in `model.relu_input_list`. 


![Activation Limiting for Resnet18 layers][limiting.png]


The following code enables the activation limiting in ecco_dnn:


```
limiting_layer_idx = [1, 4, 10, 16, 25, 31, 40, 46, 55]
optimizer = ECCO_DNN(net.parameters(), eta = args.eta, alpha_0=args.alpha_0, 
                     limiting_layer_idx=limiting_layer_idx, num_layers = 62, 
                     layer_type="bn")
```

</details>

## Citing the Codebase
Please cite this work as:


@article{fiscko2023towards,
  title={Towards Hyperparameter-Agnostic DNN Training via Dynamical System Insights},
  author={Fiscko, Carmel and Agarwal, Aayushya and Ruan, Yihan and Kar, Soummya and Pileggi, Larry and Sinopoli, Bruno},
  journal={arXiv preprint arXiv:2310.13901},
  year={2023}
}


## Authors
For questions about the code, please contact:
Aayushya Agarwal (aayushya@andrew.cmu.edu)
Yihan Ruan (yihanr@andrew.cmu.edu)

[1] George Hebrail and Alice Berard. Individual household electric power consumption data set. Aug 2012
