'''
This file generates datasets of moves and blunders of just the middle game,
just the end game, and the middle and end game together.
'''

import pandas as pd
#-----------------------------------------------------------------------

def generate_opening(input_pkl, player, blunders=False):
    df = pd.read_pickle(input_pkl)
    df = df[(df['middle_game'] == False) & (df['end_game'] == False)]

    if blunders:
        file_name = f'data/move_splits_pkl/opening_blunders/{player}_opening_blunders.pkl'
    else:
        file_name = f'data/move_splits_pkl/opening_moves/{player}_opening_moves.pkl'

    df.to_pickle(file_name)

#-----------------------------------------------------------------------

def generate_middle(input_pkl, player, blunders=False):
    df = pd.read_pickle(input_pkl)
    df = df[df['middle_game'] == True]
    
    if blunders:
        file_name = f'data/move_splits_pkl/middle_blunders/{player}_middle_blunders.pkl'
    else:
        file_name = f'data/move_splits_pkl/middle_moves/{player}_middle_moves.pkl'

    df.to_pickle(file_name)

#-----------------------------------------------------------------------

def generate_end(input_pkl, player, blunders=False):
    df = pd.read_pickle(input_pkl)
    df = df[df['end_game'] == True]
    
    if blunders:
        file_name = f'data/move_splits_pkl/end_blunders/{player}_end_blunders.pkl'
    else:
        file_name = f'data/move_splits_pkl/end_moves/{player}_end_moves.pkl'

    df.to_pickle(file_name)

#-----------------------------------------------------------------------

def generate_middle_and_end(input_pkl, player, blunders=False):
    df = pd.read_pickle(input_pkl)
    df = df[(df['middle_game'] == True) | (df['end_game'] == True)]

    if blunders:
        file_name = f'data/move_splits_pkl/middle_and_end_blunders/{player}_middle_and_end_blunders.pkl'
    else:
        file_name = f'data/move_splits_pkl/middle_and_end_moves/{player}_middle_and_end_moves.pkl'

    df.to_pickle(file_name)

#-----------------------------------------------------------------------

def main():

    top_bullet_players_list = [
        'wizard98',
        'nihalsarin2004',
        'mishka_the_great',
        'ediz_gurel',
        'rebeccaharris',
        'meneermandje',
        'night-king96',
        'muisback',
        'vincentkeymer2004',
        'zhigalko_sergei',
        'Tang'
    ]

    for player in top_bullet_players_list:
        print(f"Generating move splits for {player}...\n")

        moves_file = f'data/pkl/moves/{player}_moves.pkl'
        blunders_file = f'data/pkl/blunders/{player}_blunders.pkl'

        generate_opening(moves_file, player)
        generate_opening(blunders_file, player, blunders=True)
        generate_middle(moves_file, player)
        generate_middle(blunders_file, player, blunders=True)
        generate_end(moves_file, player)
        generate_end(blunders_file, player, blunders=True)
        generate_middle_and_end(moves_file, player)
        generate_middle_and_end(blunders_file, player, blunders=True)
        
#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()