import torch.nn as nn
import torch.nn.functional as F

class SimpleDNNUpdate(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2
                       ,hidden_size3,hidden_size,output):
        super(SimpleDNNUpdate, self).__init__()
        self.f_connected1=nn.Linear(input_size,hidden_size1)
        self.f_connected2=nn.Linear(hidden_size1,hidden_size2)
        self.f_connected3=nn.Linear(hidden_size2,hidden_size3)
        self.f_connected4=nn.Linear(hidden_size3,hidden_size)
        self.out_connected=nn.Linear(hidden_size,output)
    def forward(self,x):
        self.ec_step_input = []
        self.relu_input_list = []
        self.ec_step_input.append(x.detach().clone())

        out = self.f_connected1(x)
        self.relu_input_list.append(out.detach().clone())
        out=F.relu(out) 
        self.ec_step_input.append(out.detach().clone())

        out = self.f_connected2(out)
        self.relu_input_list.append(out.detach().clone())
        out=F.relu(out)
        self.ec_step_input.append(out.detach().clone())

        out = self.f_connected3(out)
        self.relu_input_list.append(out.detach().clone())
        out=F.relu(out)
        self.ec_step_input.append(out.detach().clone())
        
        out = self.f_connected4(out)
        self.relu_input_list.append(out.detach().clone())
        out=F.relu(out)
        out=self.out_connected(out)
        return out


# input_size=784   #28X28 pixel of image
# hidden_size1=200 #size of 1st hidden layer(number of perceptron)
# hidden_size2=150 #size of second hidden layer
# hidden_size3=100 #size of third hidden layer
# hidden_size=80   #size of fourth hidden layer
# output =10       #output layer
# bach_size=100
# net = SimpleDNNUpdate(input_size,hidden_size1,hidden_size2
#                        ,hidden_size3,hidden_size,output)
# for param in net.parameters():
#     print(param.shape)