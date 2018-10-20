import torch
import torch.nn as nn

class MyActivationFunction(nn.Module):

    def __init__(self, leak=0.01):
        super(MyActivationFunction, self).__init__()
        self.leak = leak

    def forward(self, x):
        return torch.where(x>0, x, x * self.leak)