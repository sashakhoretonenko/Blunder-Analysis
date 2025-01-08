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