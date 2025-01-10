'''
This file helps us determine which MLP model to use for our project. We test
the flexible MLP with 1, 2, and 3 affine layers, as well as with the 3 different
optimizers.
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
from models.MLPs import flexMLP  # Assuming you have a flexMLP model defined
from models.multiClassTrainer import multiTrainer
from generating_data.generate_loaders import create_Tang_MLP_loaders
#-----------------------------------------------------------------------
def train_model(aff_layers, hidden_sizes, optim_name, Tang_train_loader, Tang_val_loader, num_epochs, learning_rate, momentum, save_path, results):
    """Train a single model and store results using multiprocessing"""
    print(f"\nTraining Model: {aff_layers} Aff Layers, {optim_name}")

    # Initialize model and trainer
    model = flexMLP(num_affine_layers=aff_layers, hidden_sizes=hidden_sizes)
    trainer = multiTrainer(net=model, train_loader=Tang_train_loader, val_loader=Tang_val_loader)

    # Train model
    completed_model, best_model, train_losses, val_losses, train_acc, val_acc = trainer.train(
        optim_name=optim_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        verbose=True
    )

    # Save model and results
    model_name = f"{aff_layers}aff_{optim_name}.pt"
    save_model_path = os.path.join(save_path, model_name)
    torch.save(best_model.state_dict(), save_model_path)

    # Store results in shared dictionary
    results.append({
        'aff_layers': aff_layers,
        'optimizer': optim_name,
        'best_val_acc': max(val_acc)
    })
#-----------------------------------------------------------------------
def main():
    Tang_train_loader, Tang_val_loader, Tang_test_loader = create_Tang_MLP_loaders('data/pkl/moves/Tang_moves.pkl')

    aff_layers_options = [1, 2, 3, 4, 5]
    hidden_sizes = [128, 64, 32, 16, 8]
    optimizers = ['adamw', 'sgd', 'nesterov']
    num_epochs = 30
    learning_rate = 0.005
    momentum = 0.99
    save_path = 'savedModels/MLPgridSearch'

    manager = Manager()
    results = manager.list()
    processes = []

    for aff_layers in (aff_layers_options):
        for optim_name in optimizers:
            p = Process(target=train_model, args=(aff_layers, hidden_sizes, optim_name, Tang_train_loader, Tang_val_loader, num_epochs, learning_rate, momentum, save_path, results))
            p.start()
            processes.append(p)

    for p in processes:

        p.join()

    results_df = pd.DataFrame(list(results))
    results_df.to_excel('data/xlsx/MLP_grid_search_results.xlsx', index=False)

    print("Grid search completed. Results saved to 'grid_search_results.csv'.")
#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()