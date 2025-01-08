'''
This file helps us determine which CNN model to use for our project. We perform
a grid search over every combination of 1-3 convolutional layers and 1-2 affine
layers. In previous testing we determined that kernel size doesn't have a significant
effect on the output, so we're going to figure out the correct number of layers first
and then focus on the kernel sizes.
'''

import torch
import itertools
from multiprocessing import Process, Manager
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CNNs import flexCNN
from models.multiClassTrainer import multiTrainer
from generating_data.generate_loaders import create_Tang_multi_loaders

#-----------------------------------------------------------------------

def main():
    Tang_train_loader, Tang_val_loader, Tang_test_loader = create_Tang_multi_loaders('data/pkl/moves/Tang_moves.pkl')

    momentum = 0.99
    num_epochs = 30
    dropout = 1.0
    reg = 0.0
    learning_rate = 0.005
    save_path = "savedModels/gridSearch"

    conv_list = [1, 2, 3]
    aff_list = [1, 2, 3]
    kernel_list = [[3], [3,3], [3,3,3]]
    optim_list = ['adamw', 'sgd', 'nesterov']

    results = []

    for conv_layers, kernel_sizes in zip(conv_list, kernel_list):
        for aff_layers in aff_list:
            for optim_name in optim_list:
                print(f"\nTraining Model: {conv_layers} Conv Layers, {aff_layers} Aff Layers, {optim_name}")

                # Initialize model and trainer
                model = flexCNN(conv_layers=conv_layers, aff_layers=aff_layers, kernel_sizes=kernel_sizes)
                trainer = multiTrainer(net=model, train_loader=Tang_train_loader, val_loader=Tang_val_loader)

                # Train model
                completed_model, best_model, train_losses, val_losses, train_acc, val_acc = trainer.train(
                    optim_name=optim_name,
                    num_epochs=num_epochs,
                    dropout=dropout,
                    reg=reg,
                    momentum=momentum,
                    learning_rate=learning_rate,
                    verbose=True
                )

                # Save model and results
                model_name = f"{conv_layers}conv_{aff_layers}aff_{optim_name}.pt"
                save_model_path = os.path.join(save_path, model_name)
                torch.save(best_model.state_dict(), save_model_path)

                # Store results in the list
                results.append({
                    'conv_layers': conv_layers,
                    'aff_layers': aff_layers,
                    'kernel_sizes': kernel_sizes,
                    'optimizer': optim_name,
                    'val_acc': val_acc[-1],
                    'model_path': save_model_path
                })

                print(f"Model saved at {save_model_path} with Validation Accuracy: {val_acc[-1]:.4f}")

    # Save results summary as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_excel("data/xlsx/model_summary.xlsx", index=False)

    print("\nAll models trained and saved successfully!")

#-----------------------------------------------------------------------

if __name__ == "__main__":
    main()