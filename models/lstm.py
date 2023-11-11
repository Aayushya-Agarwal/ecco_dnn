import math

import torch
import torch.nn as nn
import torch.nn.functional as F

timesteps = 1
input_features = 6
h1_features = 200
h2_features = 200
h3_features = 1
output_features = 1

class LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=h1_features)
    self.lstm2 = torch.nn.LSTM(input_size=h1_features, hidden_size=h2_features)
    self.fc1 = torch.nn.Linear(h2_features, h3_features)

  def forward(self, inputs):
    h1, (h1_T,c1_T) = self.lstm1(inputs)
    h2, (h2_T, c2_T) = self.lstm2(h1)
    #print("inputs SHAPE: ", inputs.size())
    #print("H1 SHAPE: ", h1.size())
    #print("H2 SHAPE: ", h2.size())
    output = self.fc1(h2[:,-1,:])       # inplace of h2[-1,:,:] we can use h2_T. Both are identical
    return output

