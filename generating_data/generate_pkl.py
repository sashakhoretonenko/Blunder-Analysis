'''
This file generates games, moves, and blunders dataframes
for the top bullet players. We note that the number of
usable games actually ends up being about 25% smaller
than our original because we drop games where 2 of
the GMs play each other. Since top players tend to
play other top players, this is a significant portion.
'''

import json
import pandas as pd
import chess
import chess.pgn
import copy
#-----------------------------------------------------------------------
'''
Flips a square vertically and horizontally

Input: string of length 2 where car1 is the file and char2 is the rank
Output: string of length 2 where the flipped file and rank are returned
'''
def flip_black_square(square):
    # flip the file
    file = square[0]
    if file == 'a':
        file = 'h'
    elif file == 'b':
        file = 'g'
    elif file == 'c':
        file = 'f'
    elif file == 'd':
        file = 'e'
    elif file == 'e':
        file = 'd'
    elif file == 'f':
        file = 'c'
    elif file == 'g':
        file = 'b'
    elif file == 'h':
        file = 'a'
    
    # flip the rank
    rank = square[1]
    if rank == '1':
        rank = '8'
    elif rank == '2':
        rank = '7'
    elif rank == '3':
        rank = '6'
    elif rank == '4':
        rank = '5'
    elif rank == '5':
        rank = '4'
    elif rank == '6':
        rank = '3'
    elif rank == '7':
        rank = '2'
    elif rank == '8':
        rank = '1'
    
    return file+rank

#-----------------------------------------------------------------------

def flip_black_fen(fen):
    '''
    Split the FEN string into its components.
    This returns a list with 6 elements"
    board, active_color, castling_rights, en_passant, halfmove_clock, fullmove_number
    '''
    fen_parts = fen.split()
    board = fen_parts[0]

    # separate the board into rows
    rows = board.split('/')
    # reverse the characters in the rows (horizontal flip)
    # don't forget to reverse upper and lowercase
    for row in rows:
        for char in row:
            if char.isalpha():
                if char.islower():
                    row = row.replace(char, char.upper())
                else:
                    row = row.replace(char, char.lower())
        
        # reverse the row
        row = row[::-1]
    # reverse the rows (vertical flip)
    rows.reverse()
    # reassemble the rows into a board
    board = '/'.join(rows)
    flipped_board = ''
    for char in board:
        if char.isalpha():
            if char.islower():
                flipped_board += char.upper()
            else:
                flipped_board += char.lower()
        else:
            flipped_board += char

    # flip the active color
    if fen_parts[1] == 'w':
        active_color = 'b'
    else:
        active_color = 'w'
    
    # flip the castling rights
    castling_rights = fen_parts[2]
    if castling_rights != '-':
        for char in castling_rights:
            flipped = ""
            if char.islower():
                flipped += char.upper()
            else:
                flipped += char.lower()

            lower = []
            for char in castling_rights:
                if char.islower():
                    lower.append(char)
            upper = []
            for char in castling_rights:
                if char.isupper():
                    upper.append(char)
            castling_rights = ''.join(upper) + ''.join(lower)

    # flip the en passant squares
    en_passant = fen_parts[3]
    if en_passant != '-':
        en_passant = flip_black_square(en_passant)

    str = ' '.join([flipped_board, active_color, castling_rights, en_passant, fen_parts[4], fen_parts[5]])
    return str

#-----------------------------------------------------------------------

'''
Converts a game in pgn format to the dataframe format we want
and appends it to the games dataframe.

Input: pgn game in json format, dataframe to append to

Output: updated dataframe
'''

def parse_line(line, df, list_of_players):
    # list of top bullet players on lichess whose games we will be analyzing

    data = json.loads(line)
    # we only want standard varations          
    if data['variant'] != 'standard':
        return df
    
    # set the player data
    if data['players']['white']['user']['id'] in list_of_players:
        main_player = data['players']['white']['user']['id']
        main_color = 'white'
        opponent = data['players']['black']['user']['id']
        opponent_color = 'black'
    elif data['players']['black']['user']['id'] in list_of_players:
        main_player = data['players']['black']['user']['id']
        main_color = 'black'
        opponent = data['players']['white']['user']['id']
        opponent_color = 'white'
    else:
        print(data['players']['white']['user']['id'])
        print(data['players']['black']['user']['id'])
        print()
        return df
    
    # create row and add necessary columns
    row = {}
    row['game_id'] = data['id']
    row['speed'] = data['speed']
    row['main_player'] = main_player
    row['opponent'] = opponent
    row['main_color'] = main_color
    row['opponent_color'] = opponent_color
    row['main_rating'] = data['players'][main_color]['rating']
    row['opponent_rating'] = data['players'][opponent_color]['rating']

    moves = data['moves'].split()
    row['moves'] = moves
    row['clocks'] = data['clocks']
    row['analysis'] = data['analysis']

    if main_color == 'white':
        row['main_moves'] = moves[::2]
        row['main_clocks'] = data['clocks'][::2]
        row['main_analysis'] = data['analysis'][::2]
        row['opponent_moves'] = moves[1::2]
        row['opponent_clocks'] = data['clocks'][1::2]
        row['opponent_analysis'] = data['analysis'][1::2]

    else:
        row['main_moves'] = moves[1::2]
        row['main_clocks'] = data['clocks'][1::2]
        row['main_analysis'] = data['analysis'][1::2]
        row['opponent_moves'] = moves[::2]
        row['opponent_clocks'] = data['clocks'][::2]
        row['opponent_analysis'] = data['analysis'][::2]

    # get winner
    if data['status'] != 'draw' and data['status'] != 'stalemate':
        if row['main_clocks'][len(row['main_clocks']) - 1] == 0:
            row['winner'] = data['players'][opponent_color]['user']['name']
        elif row['opponent_clocks'][len(row['opponent_clocks']) - 1] == 0:
            row['winner'] = main_player
        elif data['winner']== main_color:
            row['winner'] = main_player
        else:
            row['winner'] = data['players'][opponent_color]['user']['name']

    row['status'] = data['status']
    row['clock_initial'] = data['clock']['initial'] if 'clock' in data else None
    row['clock_increment'] = data['clock']['increment'] if 'clock' in data else None
    row['middle_game_begins'] = data['division']['middle'] if 'middle' in data['division'] else None
    row['end_game_begins'] = data['division']['end'] if 'end' in data['division'] else None

        
    # these columns probably aren't important but I'll include them in the database just in case
    row['opening_eco'] = data['opening']['eco']
    row['opening_name'] = data['opening']['name']

    temp_df = pd.DataFrame([row])
    df = pd.concat([df, temp_df], ignore_index=True)
    return df

#-----------------------------------------------------------------------

'''
Takes a game from the dataframe and extracts the moves from it

Input: row from the games dataframe
Output: dataframe of moves from that game

'''
def extract_moves(row):
    starting_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    board = chess.Board(fen=starting_FEN)
    moves = row['moves']

    # analysis stops after 200 moves
    if(len(moves) > 200):
        return None

    clocks_copy = copy.deepcopy(row['clocks'])
    analysis_copy = copy.deepcopy(row['analysis'])

    status = row['status'] 
    if status != 'mate':
        clocks_copy.pop()
    else:
        analysis_copy.append({'mate': 0})

    # dataframe structure just so I can remember
    columns = [
        'game_id', 'move_id', 'move_number', 'player_color', 'player_moving', 'move', 'previous_move'
        'eval', 'mate', 'category', 'best', 'blunder', 'clock_time',
        'before_fen', 'after_fen', 'middle_game', 'end_game'
    ]

    total_moves_df = pd.DataFrame()
    previous_move = None
    move_counter = 1

    for move in moves:
        x = {}
        try:
            before_fen = board.fen()
            board.push_san(move)
            after_fen = board.fen()
        except ValueError:
            print(f"Invalid move {move} for board at move {move_counter}. Skipping.")
            return None

        # store data in dataframe
        x['game_id'] = row['game_id']
        x['move_id'] = row['game_id'] + move
        x['move_number'] = move_counter
        if move_counter % 2 == 1:       # note that we count moves as plies
            x['player_color'] = 'white'
        else:
            x['player_color'] = 'black'
        if x['player_color'] == row['main_color']:
            x['moving_player'] = row['main_player']
        else:
            x['moving_player'] = row['opponent']
        x['move'] = move
        # store the previous move and set the new previous move to be the current move
        x['previous_move'] = previous_move
        previous_move = move
        x['eval'] = analysis_copy[move_counter - 1]['eval'] if 'eval' in analysis_copy[move_counter - 1] else None
        x['category'] = analysis_copy[move_counter - 1]['judgment']['name'] if 'judgment' in analysis_copy[move_counter - 1] else None
        x['best'] = analysis_copy[move_counter - 1]['best'] if 'best' in analysis_copy[move_counter - 1] else None
        x['blunder'] = x['category'] == 'Blunder'

        x['before_fen'] = before_fen
        x['after_fen'] = after_fen
        x['middle_game'] = False
        if row['middle_game_begins'] is not None:
            if move_counter >= row['middle_game_begins']:
                x['middle_game'] = True
        x['end_game'] = False
        if row['end_game_begins'] is not None:
            if move_counter > row['end_game_begins']:
                x['middle_game'] = False
                x['end_game'] = True

        # append move to total_moves_df
        temp_df = pd.DataFrame([x])
        total_moves_df = pd.concat([total_moves_df, temp_df], ignore_index=True)

        move_counter += 1

    return total_moves_df

#-----------------------------------------------------------------------

'''
Generates a dataframe of games from an ndjson file in the format we want

Input: ndjson file, output pkl file
Output: None
'''
def generate_games_pkl(input_file, output_pkl, list_of_players):
    games_df = pd.DataFrame()

    with open(input_file, 'r') as file:
        for line in file:
            # load_singular_line_to_df takes a dataframe and adds to it from the rest of the pgn file
            games_df = parse_line(line, games_df, list_of_players=list_of_players)

    print("Total games before filtering: ", len(games_df))

    # remove duplicate game_ids just in case
    games_df = games_df.drop_duplicates(subset='game_id', keep='first')
    print("Total games after filtering: ", len(games_df))

    # save the games dataframe to a pickle file
    games_df.to_pickle(output_pkl)

    return games_df

#-----------------------------------------------------------------------
'''
Generates a dataframe of moves from a games dataframe

Input: games dataframe, output pkl file
Output: None
'''
def generate_moves_pkl(input_file, output_pkl, list_of_players):
    df = pd.read_pickle(input_file)
    moves_df = pd.DataFrame()
    num_iters = len(df)

    for i in range(num_iters):
        row = df.iloc[i]
        if row is not None:
            temp_df = extract_moves(row)
            moves_df = pd.concat([moves_df, temp_df], ignore_index=True)
        if i % 500 == 0 and i > 0: 
            print(f"Processed {i} games")

    print("\nProcessed all moves. Total moves before filtering: ", len(moves_df))

    # get only the moves of the top bullet players
    top_moves_df = moves_df[moves_df['moving_player'].isin(list_of_players)]
    print("Total moves after filtering: ", len(top_moves_df))
    print()

    # lastly, we need to flip the board for the black player to normalize the data
    def flip(row):
        if row['player_color'] == 'black':
            row['before_fen'] = flip_black_fen(row['before_fen'])
            row['after_fen'] = flip_black_fen(row['after_fen'])
        return row
    
    top_moves_df = top_moves_df.apply(flip, axis=1)

    # save the moves dataframe to a pickle file
    top_moves_df.to_pickle(output_pkl)

    return top_moves_df
#-----------------------------------------------------------------------

'''
Generates a dataframe of blunders from a moves dataframe

Input: moves dataframe, output pkl file
Output: None
'''
def generate_blunders_pkl(input_pkl, output_pkl):
    df = pd.read_pickle(input_pkl)
    blunders_df = df[df['blunder'] == True]
    blunders_df.to_pickle(output_pkl)

    return blunders_df

#-----------------------------------------------------------------------

def main():
    # list of top bullet players on lichess whose games we will be analyzing
    top_bullet_players_list = [
        # 'wizard98',
        # 'nihalsarin2004',
        'mishka_the_great',
        # 'ediz_gurel',
        # 'rebeccaharris',
        # 'meneermandje',
        # 'night-king96',
        # 'muisback',
        # 'vincentkeymer2004',
        # 'zhigalko_sergei'
    ]

    for player in top_bullet_players_list:
        player_list = [player]

        file_name = f'data/ndjson/{player}_games.ndjson'
        games_pkl = f'data/pkl/games/{player}_games.pkl'
        moves_pkl = f'data/pkl/moves/{player}_moves.pkl'
        blunders_pkl = f'data/pkl/blunders/{player}_blunders.pkl'

        print(f"Generating games dataframe for {player}...")
        games_df = generate_games_pkl(input_file=file_name, output_pkl=games_pkl, list_of_players=player_list)
        print(f"Length of {player} games dataframe: {len(games_df)}\n")

        print(f"Generating moves dataframe for {player}...")
        moves_df = generate_moves_pkl(games_pkl, output_pkl=moves_pkl, list_of_players=player_list)
        print(f"Length of {player} moves dataframe: {len(moves_df)}\n")

        print(f"Generating blunders dataframe for {player}...")
        blunders_df = generate_blunders_pkl(moves_pkl, output_pkl=blunders_pkl)
        print(f"Length of {player} blunders dataframe: {len(blunders_df)}\n")


    # # gets the data for Tang specifically
    # Tang_list = ['penguingm1', 'penguingim1']

    # file_name = 'data/ndjson/Tang_games.ndjson'
    # games_pkl = 'data/pkl/games/Tang_games.pkl'
    # moves_pkl = 'data/pkl/moves/Tang_moves.pkl'
    # blunders_pkl = 'data/pkl/blunders/Tang_blunders.pkl'

    # print("Generating games dataframe for Tang...")
    # Tang_games_df = generate_games_pkl(input_file=file_name, output_pkl=games_pkl, list_of_players=Tang_list)
    # print(f"Length of Tang games dataframe: {len(Tang_games_df)}\n")

    # print("Generating moves dataframe for Tang...")
    # Tang_moves_df = generate_moves_pkl(games_pkl, output_pkl=moves_pkl, list_of_players=Tang_list)
    # print(f"Length of Tang moves dataframe: {len(Tang_moves_df)}\n")

    # print("Generating blunders dataframe for Tang...")
    # Tang_blunders_df = generate_blunders_pkl(moves_pkl, output_pkl=blunders_pkl)
    # print(f"Length of Tang blunders dataframe: {len(Tang_blunders_df)}\n")

#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()
