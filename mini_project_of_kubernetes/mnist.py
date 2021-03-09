import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.W = nn.Parameter(torch.randn(784, 10))
        self.b = nn.Parameter(torch.zeros([10]))

    def forward(self, x):
        x = self.flatten(x)
        y = torch.matmul(x, self.W) + self.b
        # Applies the Softmax function to an n-dimensional 
        # input Tensor rescaling them so that the elements 
        # of the n-dimensional output Tensor lie in the range 
        # [0,1] and sum to 1.
        y = nn.Softmax(dim=1)(y)
        return y