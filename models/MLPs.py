'''
This file defines the different CNN models that are used in the project,
as well as the trainer class that trains them. We note that the trainer
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------
'''
A flexible MLP class that allows you to specify the number of affine layers,
as well as the output sizes of the hidden layers.

Input:
- hidden_sizes: list of sizes for each hidden layer
- num_affine_layers: number of affine layers
- input_size: size of the input tensor
- output_size: size of the output tensor
'''

class flexMLP(nn.Module):
    def __init__(self, hidden_sizes, num_affine_layers, input_size=384, output_size=6):
        super(flexMLP, self).__init__()
        
        # Validate the number of layers
        if num_affine_layers < 1 or num_affine_layers > 3:
            raise ValueError("num_affine_layers must be between 1 and 3")
        
        self.num_affine_layers = num_affine_layers
        self.relu = nn.ReLU()
        
        if num_affine_layers == 1:
            self.fc1 = nn.Linear(input_size, output_size)
        elif num_affine_layers == 2:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], output_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        
        # Apply the layers based on the specified number
        x = self.relu(self.fc1(x))
        
        if self.num_affine_layers > 1:
            x = self.relu(self.fc2(x))
        
        if self.num_affine_layers == 3:
            x = self.fc3(x)
        
        return x