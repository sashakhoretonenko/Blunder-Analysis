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
        self.hidden_sizes = hidden_sizes
        self.num_affine_layers = num_affine_layers
        self.aff_layers = nn.ModuleList()

        # Validate the number of layers
        if num_affine_layers < 1 or num_affine_layers > 5:
            raise ValueError("num_affine_layers must be between 1 and 5")
        
        self.relu = nn.ReLU()

        # Dynamically create affine layers
        in_size = input_size
        for i in range(num_affine_layers - 1):
            self.aff_layers.append(nn.Linear(in_size, hidden_sizes[i]))
            in_size = hidden_sizes[i]
        # Final layer mapping to output_size
        self.aff_layers.append(nn.Linear(in_size, output_size))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        
        # Apply the affine layers
        for i in range(len(self.aff_layers) - 1):
            x = self.relu(self.aff_layers[i](x))
        # No ReLU activation on the final layer
        x = self.aff_layers[-1](x)
        
        return x