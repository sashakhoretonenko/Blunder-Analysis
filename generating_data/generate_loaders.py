'''
This file is the final step in generating data that we can use in
our neural networks. It takes the data from the pkl files and converts
it into usable tensors.
'''

import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


def extract_piece_type(move):
    # Check for castling moves first
    if move in ['O-O', 'O-O-O']:
        return 'K'
    elif move.startswith(('K', 'Q', 'R', 'B', 'N')):
        return move[0]
    else:
        return 'P'

#-----------------------------------------------------------------------

piece_to_label = {
    'K': 0,
    'Q': 1,
    'R': 2,
    'B': 3,
    'N': 4,
    'P': 5
}

#-----------------------------------------------------------------------
'''
Converts a FEN string to a 384 length 1 dimensional tensor that 
represents the board state.

Input: FEN string
Output: 6x8x8 tensor
'''

def fen_to_384tensor(fen):
    fen_items = fen.split()
    board = fen_items[0]
    rows = board.split('/')
    tensor_384 = torch.zeros(384)

    piece_to_channel = {
        'P': (0, 1),  
        'p': (0, -1), 
        'N': (1, 1),  
        'n': (1, -1),
        'B': (2, 1),  
        'b': (2, -1),
        'R': (3, 1),  
        'r': (3, -1),
        'Q': (4, 1),  
        'q': (4, -1),
        'K': (5, 1),  
        'k': (5, -1)
    }

    for i, row in enumerate(rows):
        if row == '8':
            continue
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char) # skip the empty squares
            else:
                piece = char
                skip, value = piece_to_channel[piece]
                tensor_384[64*skip + 8*i + col_idx] = value
    
    return tensor_384


#-----------------------------------------------------------------------
'''
Converts a FEN string to a 6x8x8 tensor that represents the board state.

Input: FEN string
Output: 6x8x8 tensor
'''

def fen_to_tensor(fen):
    fen_items = fen.split()

    piece_to_channel = {
        'P': (0, 1),  
        'p': (0, -1), 
        'N': (1, 1),  
        'n': (1, -1),
        'B': (2, 1),  
        'b': (2, -1),
        'R': (3, 1),  
        'r': (3, -1),
        'Q': (4, 1),  
        'q': (4, -1),
        'K': (5, 1),  
        'k': (5, -1),
    }

    board_tensor = torch.zeros((6, 8, 8), dtype=torch.float32)
    board_fen = fen_items[0]
    rows = board_fen.split('/')

    for i, row in enumerate(rows):
        # skip the row if it is empty, no updates needed
        if row == '8':
            continue
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char) # skip the empty squares
            else:
                channel, value = piece_to_channel[char] # get channel and value
                board_tensor[channel, i, col_idx] = value
                col_idx += 1

    return board_tensor

#-----------------------------------------------------------------------

'''
Converts a dataframe into 6x8x8 tensors that
represent the board states.

Input: dataframe
Output: multiDataset object
'''

class multiDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        before_fen = self.dataframe.iloc[idx]['before_fen']
        tensor = fen_to_tensor(before_fen)
        piece_type = self.dataframe.iloc[idx]['piece_type']
        label = piece_to_label[piece_type]
        label_tensor = torch.tensor(label, dtype=torch.long) 
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label_tensor
    
#-----------------------------------------------------------------------
'''
Converts a dataframe into a length 384 tensor that represents the board state.

Input: dataframe
Output: MLPDataset object
'''

class MLPDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        before_fen = self.dataframe.iloc[idx]['before_fen']
        tensor = fen_to_384tensor(before_fen)
        piece_type = self.dataframe.iloc[idx]['piece_type']
        label = piece_to_label[piece_type]
        label_tensor = torch.tensor(label, dtype=torch.long) 
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label_tensor
    
#-----------------------------------------------------------------------
'''
Creates train, validation, and test loaders for using the MLP model.
This function is only used for Tang's data, as the data of the other
grandmasters is only test data.

Input: pkl file of Tang's moves
Output: train, validation, and test loaders
'''

def create_Tang_MLP_loaders(pkl_file, batch_size=256):
    Tang_df = pd.read_pickle(pkl_file)

    # We use random_state=397 to ensure reproducibility and because we love COS 397!!

    # Train on 70% of the data
    train_df, test_and_val_df = train_test_split(Tang_df, test_size=0.3, random_state=397)
    # Validate and test on 15% of the data each
    test_df, val_df = train_test_split(test_and_val_df, test_size=0.5, random_state=397)

    train_dataset = MLPDataset(train_df)
    test_dataset = MLPDataset(test_df)
    val_dataset = MLPDataset(val_df)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
#-----------------------------------------------------------------------
'''
Creates just a test loader out of the GM data

Input: pkl file of GM moves
Output: test loader
'''

def create_GM_MLP_loader(pkl_file, batch_size=256):
    GM_df = pd.read_pickle(pkl_file)
    test_dataset = MLPDataset(GM_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader
#-----------------------------------------------------------------------
'''
Creates train, validation, and test loaders for using the CNN model.
We note that multiDataset should be renamed to MLPDataset, but we
are going to have to clean that up later. This function is only used
for Tang's data, as the data of the other grandmasters is only test data.

Input: pkl file of Tang's moves
Output: train, validation, and test loaders
'''

def create_Tang_multi_loaders(pkl_file, batch_size=256):
    Tang_df = pd.read_pickle(pkl_file)
    
    # We use random_state=397 to ensure reproducibility and because we love COS 397!!

    # Train on 70% of the data
    train_df, test_and_val_df = train_test_split(Tang_df, test_size=0.3, random_state=397)
    # Validate and test on 15% of the data each
    test_df, val_df = train_test_split(test_and_val_df, test_size=0.5, random_state=397)

    train_dataset = multiDataset(train_df)
    test_dataset = multiDataset(test_df)
    val_dataset = multiDataset(val_df)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

#-----------------------------------------------------------------------
'''
Creates just a test loader out of the GM data

Input: pkl file of GM moves
Output: test loader
'''

def create_GM_test_multi_loaders(pkl_file, batch_size=256):
    GM_df = pd.read_pickle(pkl_file)
    test_dataset = multiDataset(GM_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader
