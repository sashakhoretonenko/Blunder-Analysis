'''
This file defines the flexCNN model that is used in the project,
which is a flexible CNN model that allows you to specify the number
of convolutional and affine layers, as well as the kernel sizes.
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
CNN model with 2 convolutional layers and 1 fully connected layer.
All convolutional layers have kernels of size 3. While we can just use
flexCNN, it can sometime be annoying with the model loading, so we've
created this backup just in case.
'''

class CNN_2convlayer_1afflayer(nn.Module):
    def __init__(self):
        super(CNN_2convlayer_1afflayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        return x

#-----------------------------------------------------------------------

'''
A flexible CNN class that allows you to specify the number of convolutional,
layers, affine layers, and the kernel sizes.

Input:
- conv_layers: number of convolutional layers
- aff_layers: number of affine layers
- kernel_sizes: list of kernel sizes for each convolutional layer
'''
class flexCNN(nn.Module):
    def __init__(self, conv_layers=1, aff_layers=1, kernel_sizes=None):

        super(flexCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.aff_layers = nn.ModuleList()

        # Default to kernel sizes of size 3
        if kernel_sizes is None:
            kernel_sizes = [3] * conv_layers
        else:
            conv_layers = len(kernel_sizes)

        # We start with 6 input channels
        cur_channels = 6
        # The size of our board is originally 8x8
        board_size = 8

        for i in range(conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels=cur_channels, out_channels=32*(2**i), kernel_size=kernel_sizes[i]))
            cur_channels = 32 * (2**i)
            board_size -= (kernel_sizes[i] - 1)

        # Compute the length of the flattened tensor
        flat_length = cur_channels * (board_size ** 2)

        # if we have only one affine layer, immediately output predictions
        if aff_layers == 1:
            self.aff_layers.append(nn.Linear(flat_length, 6))
        elif aff_layers == 2:
            self.aff_layers.append(nn.Linear(flat_length, 128))
            self.aff_layers.append(nn.Linear(128, 6))
        elif aff_layers == 3:
            self.aff_layers.append(nn.Linear(flat_length, 256))
            self.aff_layers.append(nn.Linear(256, 128))
            self.aff_layers.append(nn.Linear(128, 6))

    def forward(self, x):
        # Convolutional layers with ReLU activations
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)

        # Add ReLU activations in between each layer
        for i, fc in enumerate(self.aff_layers):
            x = fc(x)
            if i < len(self.aff_layers) - 1:
                x = F.relu(x)
        return x
            
#-----------------------------------------------------------------------
