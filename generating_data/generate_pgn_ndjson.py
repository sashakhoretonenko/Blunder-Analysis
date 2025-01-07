'''
This file makes API calls to the Lichess API to fetch bullet games
for 10 of the top bullet players in the world, as well as Andrew Tang.
I make sure to fetch only games for which analysis is available.

I extract 2000 games from each of the 10 players who I selected
(each of these 10 players are top bullet players in the world).
I also extract 10000 games from Andrew Tang, as he is the player
whose game I want to focus on.
'''

# import statements
import requests
import json
from dotenv import load_dotenv
import os

#-----------------------------------------------------------------------
'''
Fetches bulletgames for a specific user from the Lichess API.

Input: 
'''
def fetch_bullet_games(username, max_games, headers):
    GAMES_EXPORT_URL = f"https://lichess.org/api/games/user/{username}"
    url = GAMES_EXPORT_URL.format(username=username)

    all_games = []
    games_fetched = 0

    while games_fetched < max_games:
        params = {
                'max': max_games,
                'perfType': 'bullet',
                'rated': True,
                'pgnInJson': False,
                'clocks': True,
                'evals': True,
                'accuracy': True,
                'opening': True,
                'division': True,
                'literate': True,
                'analysed': True
            }
        
        response = requests.get(GAMES_EXPORT_URL, headers=headers, params=params, stream=True)
        response.raise_for_status() # check if response was successful

        batch_games = []

        # decodes games line by line
        for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        game = line.decode("utf-8")
                        data = json.loads(game)
                        batch_games.append(game)
                        if len(batch_games) % 500 == 0:
                            print(f"Fetched {len(batch_games)} games for user {username}.")
                    except json.JSONDecodeError:
                        print(f"Warning: Couldn't parse game #{len(batch_games)}.")
                        continue

        if not batch_games:
            print("No more games available")
            break
    
        # reduce the batch of games to max_games if it's over max_games
        if len(batch_games) > max_games - games_fetched:
            batch_games = batch_games[:max_games - games_fetched]
    
        games_fetched += len(batch_games)
        all_games.extend(batch_games)
        print(f"Fetched {games_fetched} for user {username}.")


        if games_fetched >= max_games:
            print(f"Reached target number of games! for {username}\n")
            break
    
    return all_games

#-----------------------------------------------------------------------

def main():
    load_dotenv()
    API_TOKEN = os.getenv("API_TOKEN")
    HEADERS = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Accept': 'application/x-ndjson',  # Streaming ndjson format
    }

    # First we'll get the data for our 10 selected bullet players

    top_bullet_players_list = [
        'wizard98',
        # 'nihalsarin2004',
        # 'mishka_the_great',
        # 'ediz_gurel',
        # 'rebeccaharris',
        # 'meneermandje',
        # 'night-king96',
        # 'muisback',
        # 'vincentkeymer2004',
        # 'zhigalko_sergei'
    ]

    max_GM_games = 2000

    for player in top_bullet_players_list:
        username = player
        filename = f'data/ndjson/{username}_games.ndjson'
        print(f"Starting to fetch bullet games for {username}...\n")
        games = fetch_bullet_games(username, max_GM_games, headers=HEADERS)
        print(f"\nFinished fetching {len(games)} bullet games for {username}.\n")

        with open(filename, 'w') as f:
            for game in games:
                f.write(game)
                f.write('\n')


    # # Next we'll get the data for Tang specifically
    # Tang_username = 'penguingim1'
    # # We want to get 10000 of Tang's games since he is the player the model is training on
    # max_Tang_games = 9900
    # Tang_filename = 'data/ndjson/Tang_games.ndjson'
    # print(f"Starting to fetch bullet games for {Tang_username}...\n")
    # Tang_games = fetch_bullet_games(Tang_username, max_Tang_games, headers=HEADERS)
    # print(f"\nFinished fetching {len(Tang_games)} bullet games for {Tang_username}.\n")

    # with open(Tang_filename, 'w') as f:
    #     for game in Tang_games:
    #         f.write(game)
    #         f.write('\n')    

    # print("All games fetched successfully!")
   
#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()
