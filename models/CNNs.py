'''
This file defines the different CNN models that are used in the project,
as well as the trainer class that trains them. We note that the trainer

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import chess
import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------
'''
CNN model with 1 convolutional layer and 1 fully connected layer.
All convolutional layers have kernels of size 3.
'''

class CNN_1convlayer_k3_1afflayer(nn.Module):
    def __init__(self):
        super(CNN_1convlayer_k3_1afflayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 6 * 6, 6)  # Direct output after flattening

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)  
        return x

#-----------------------------------------------------------------------
'''
CNN model with 1 convolutional layer and 2 fully connected layers.
All convolutional layers have kernels of size 3.
'''

class CNN_1convlayer_k3_2afflayer(nn.Module):
    def __init__(self):
        super(CNN_1convlayer_k3_2afflayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#-----------------------------------------------------------------------


'''
CNN model with 3 convolutional layers and 2 fully connected layers.
All convolutional layers have kernels of size 3.
'''

class CNN_3convlayer_k333_2afflayer(nn.Module):
    def __init__(self):
        super(CNN_3convlayer_k333_2afflayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # flatten tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#-----------------------------------------------------------------------
'''
Function that compares whether two models have the same parameters.
'''
def compare_models(model1, model2):
    return all(torch.equal(param1, param2) for param1, param2 in zip(model1.state_dict().values(), model2.state_dict().values()))

#-----------------------------------------------------------------------
'''
Class to train the model, periodically checking the accuracy on the validation set.


'''


class multiTrainer():
    def __init__(self, net=None, train_loader=None, val_loader=None):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_state = None
        self.best_val_acc = 0.0

        # Set the device to the MAC GPU if avaialable
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.net.to(self.device)

    def train(self, optim_name, loss_function, num_epochs, 
              reg=0.0, dropout=1.0, learning_rate=1e-2, 
              momentum=0, learning_rate_decay=0.95,step_size=1,
              update='momentum', batch_size=100, acc_frequency=None, 
              verbose=False):
        
        train_losses = []
        val_losses = []
        train_acc_history = []
        val_acc_history = []

        if optim_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=0.01)
        elif optim_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)
        elif optim_name.lower() == 'nesterov':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
        else:
            raise ValueError(f"Optimizer '{optim_name}' not recognized!")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=learning_rate_decay)

        for epoch in range(num_epochs):

            if dropout < 1.0:
                self.net.train()    # dropout only during training

            epoch_train_loss = 0
            correct_train = 0   # total correct predictions
            total_train = 0     # total samples processed
            
            for data in tqdm.tqdm(self.train_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self.net(X)
                # Compute loss
                loss = loss_function(outputs, y)
                # Apply regularization if needed
                if reg > 0:
                    decay_penalty = sum(p.pow(2.0).sum() for p in self.net.parameters())
                    loss += reg * decay_penalty

                # Backpropagation
                loss.backward()
                # Optimization step
                optimizer.step()

                # Loss and accuracy tracking
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == y).sum().item()
                total_train += y.size(0)

            # Training loss and accuracy for epoch
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_acc = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_acc_history.append(train_acc)
            # --------------------------------------------------
            # Validation Phase
            self.net.eval()
            epoch_val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():  # no gradients during validation
                for data in self.val_loader:
                    X, y = data[0].to(self.device), data[1].to(self.device)

                    # Forward pass
                    outputs = self.net(X)
                    # Compute loss
                    loss = loss_function(outputs, y)
                    outputs = self.net(X)
                    loss = loss_function(outputs, y)
                    epoch_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == y).sum().item()
                    total_val += y.size(0)

            # Compute average validatoin loss
            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_acc = correct_val / total_val
            val_losses.append(avg_val_loss)
            val_acc_history.append(val_acc)

            # Save best validation model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.net.state_dict().copy()

            # Learning rate decay
            scheduler.step()

            if verbose or epoch % acc_frequency == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Return both final model and best performing model
        completed_model = self.net
        best_model = self.net.__class__()  # Create a new instance of the model
        best_model.load_state_dict(self.best_model_state)

        print(f"\nTraining completed! Best Validation Accuracy: {self.best_val_acc:.4f}")


        return completed_model, best_model, train_losses, val_losses, train_acc_history, val_acc_history
