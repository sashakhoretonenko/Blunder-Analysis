'''
This file contains the trainer that we use both for the CNN and MLP models.
'''
#-----------------------------------------------------------------------
'''
Function that compares whether two models have the same exact parameters.

Input: model1, model2
Output: boolean
'''

def compare_models(model1, model2):
    # Both models need to be on same device
    device = next(model1.parameters()).device
    model2 = model2.to(device)
    
    return all(torch.equal(param1, param2) 
               for param1, param2 in zip(model1.state_dict().values(), model2.state_dict().values()))

#-----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CNNs import flexCNN

class multiTrainer():
    def __init__(self, net=None, train_loader=None, val_loader=None):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_state = None
        self.best_val_acc = 0.0

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.net.to(self.device)

    def train(self, optim_name, num_epochs, loss_function = nn.CrossEntropyLoss(),
              reg=0.0, dropout=1.0, learning_rate=1e-2, 
              momentum=0, learning_rate_decay=0.95,step_size=1,
                acc_frequency=1, verbose=False):
        
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
                self.best_model_state = self.net.state_dict()  # Save only the state_dict
                self.best_model_config = {                     # Save the model architecture separately
                    'conv_layers': len(self.net.conv_layers),
                    'aff_layers': len(self.net.aff_layers),
                    'kernel_sizes': [layer.kernel_size[0] for layer in self.net.conv_layers]
                }
            print(f"New best model saved at epoch {epoch + 1}")

            # Learning rate decay
            scheduler.step()

            if verbose or epoch % acc_frequency == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Return both final model and best performing model
        completed_model = self.net
        # Ensure the model is reconstructed with the same architecture before loading weights
        best_model = flexCNN(
            conv_layers=self.best_model_config['conv_layers'],
            aff_layers=self.best_model_config['aff_layers'],
            kernel_sizes=self.best_model_config['kernel_sizes'])
        best_model.load_state_dict(self.best_model_state)

        print(f"\nTraining completed! Best Validation Accuracy: {self.best_val_acc:.4f}")

        if(compare_models(completed_model, best_model)):
            print("\nThe best model is the final model.")
        else:
            print(f"\nThe best model occured at epoch {val_acc_history.index(self.best_val_acc) + 1}.")

        return completed_model, best_model, train_losses, val_losses, train_acc_history, val_acc_history
