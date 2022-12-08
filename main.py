import os
import shutil
import logging

import chess
import chess.engine

from neural_net_engine import model, environment
from algorithmic_engine import Engines_and_Simulator

# Play game
def play(log_result_dir = './_logs'):
    # Load chess environment
    env = environment.ChessEnv()

    # Load Algorithmic engine
    depth = 3
    checkmateVal = 1000
    mode = 'final_minimax'
        # CCCR, greedyMove, twomove_minimax, basic_minimax, improved_minimax, final_minimax

    # Load DQN chess engine
    version = 'v0.2'
    dqn_engine = model.DQN_Agent()
    dqn_engine.load(f'./_saved_models/{version}')

    # Load stockfish engine
    stockfish_path = shutil.which('stockfish')
    if stockfish_path is None:
        raise ValueError('Could not find stockfish engine!')
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Ask the user what mode to play
    print('Options')
    print('\t1) human')
    print('\t2) Algorithmic chess engine')
    print('\t3) DQN chess engine')
    print('\t4) Stockfish chess engine')
    print('\t5) Random moves chess engine')
    player1 = int(input('Choose player 1: '))
    player2 = int(input('Choose player 2: '))
    num_of_games = int(input('How many games do you want to play?: '))

    # Start logging game results
    if player1 not in [1, 2, 3, 4, 5] or player2 not in [1, 2, 3, 4, 5]:
        raise ValueError('Please select valid players')
    else:
        if player1 == 1:
            white_player = 'Human'
        elif player1 == 2:
            white_player = f'Algorithmic_{mode}'
        elif player1 == 3:
            white_player = f'DQN_{version}'
        elif player1 == 5:
            white_player = 'RandomMove'
        else:
            white_player = 'Stockfish'

        if player2 == 1:
            black_player = 'Human'
        elif player2 == 2:
            black_player = f'Algorithmic_{mode}'
        elif player2 == 3:
            black_player = f'DQN_{version}'
        elif player2 == 5:
            black_player = 'RandomMove'
        else:
            black_player = 'Stockfish'

        # Config for logging
        if not os.path.exists(log_result_dir):
            os.makedirs(log_result_dir)

        logging.basicConfig(
                filename = os.path.join(log_result_dir, f'{white_player}-{black_player}.log'),
                filemode = 'a',
                format='%(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO
        )
        logging.info('Game_#:winner,result,reason')


    # Start the game
    for game_count in range(1, num_of_games + 1):
        # Start the game
        print('======== Starting a new game: ========')
        state = env.reset()
        print(env.render())
        print()
        
        while not env.game_over(): 
            if env._board.turn:
                # whites turn
                print('Whites move:')
                if player1 == 2:
                    if mode == 'CCCR':
                        move = Engines_and_Simulator.greedyMove(env._board.copy())
                    elif mode == 'greedyMove':
                        move = Engines_and_Simulator.greedyMove(env._board.copy())
                    elif mode == 'twomove_minimax':
                        move = Engines_and_Simulator.twomove_minimax(env._board.copy())
                    elif mode == 'basic_minimax':
                        move = Engines_and_Simulator.basic_minimax(env._board.copy(), depth)[1]
                    elif mode == 'improved_minimax':
                        move = Engines_and_Simulator.improved_minimax(env._board.copy(), depth, -checkmateVal, checkmateVal)[1]
                    elif mode == 'final_minimax':
                        checkmateVal, _, move = Engines_and_Simulator.final_minimax(env._board.copy(), depth, -checkmateVal, checkmateVal, True)
                    env.play(move)
                elif player1 == 3: 
                    # prediction done by q-network
                    action = dqn_engine.predict(state, env)
                    env.step(action)
                elif player1 == 5:
                    move = Engines_and_Simulator.randomMove(env._board.copy())
                    env.play(move)
                else:
                    move = stockfish_engine.play(env._board, chess.engine.Limit(time = 0.01)).move
                    env.play(move)
            else:
                # blacks turn
                print('Blacks move:')
                if player2 == 2:
                    if mode == 'CCCR':
                        move = Engines_and_Simulator.greedyMove(env._board.copy())
                    elif mode == 'greedyMove':
                        move = Engines_and_Simulator.greedyMove(env._board.copy())
                    elif mode == 'twomove_minimax':
                        move = Engines_and_Simulator.twomove_minimax(env._board.copy())
                    elif mode == 'basic_minimax':
                        move = Engines_and_Simulator.basic_minimax(env._board.copy(), depth)[1]
                    elif mode == 'improved_minimax':
                        move = Engines_and_Simulator.improved_minimax(env._board.copy(), depth, -checkmateVal, checkmateVal)[1]
                    elif mode == 'final_minimax':
                        checkmateVal, _, move = Engines_and_Simulator.final_minimax(env._board.copy(), depth, -checkmateVal, checkmateVal, False)
                    env.play(move)
                elif player2 == 3: 
                    # prediction done by q-network
                    action = dqn_engine.predict(state, env)
                    env.step(action)
                elif player2 == 5:
                    move = Engines_and_Simulator.randomMove(env._board.copy())
                    env.play(move)
                else:
                    move = stockfish_engine.play(env._board, chess.engine.Limit(time = 0.01)).move
                    env.play(move)

            # move & print current board
            print(env.render())
            print()

        # Show result of game & log game information into file
        env.result()
        env.logResult(game_count)
        print('============ Ending game ============')

if __name__ == '__main__':
    play(log_result_dir='./_logs')


