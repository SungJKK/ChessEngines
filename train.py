import os
import logging
import shutil

import chess
import chess.engine

from neural_net_engine import model, environment
from algorithmic_engine import Engines_and_Simulator


def train(batch_size: int = 32, epochs: int = 1000, version: str = 'v0.0'):
    '''
    @params:
        batch_size: int = batch size for gradient descent
        epochs: int = number of games for our agent to play
        timestep_per_epoch: int = if game doesn't end, maximum game time (moves) to train)
    '''
    # load chess environment
    env = environment.ChessEnv()

    # load dqn engine
    agent = model.DQN_Agent()

    # run game
    timestep_per_epoch = 250
    for epoch in range(0, epochs + 1):
        # reset the environment
        state = env.reset()

        for timestep in range(0, timestep_per_epoch):
            # run action by agent
            action = agent.act(state, env)
            
            # take action by environment
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                print(f'Episode: {epoch}/{epochs}, score: {timestep}, e: {agent.epsilon:.2}')
                env.logTraining(f'{epoch}/{epochs}', timestep, f'{agent.epsilon:.6}')
                    # log exploration vs. exploitation rate over time 
                    # if our agent is not performing well, good place to look is the epsilon
                break

        if batch_size < len(agent.experience_replay):
            agent.replay(batch_size)


        # save the model weights every N games
        if epoch % 100 == 0:
            agent.align_target_model()
            agent.save(f'./_saved_models/{version}')


if __name__ == '__main__':
    # training version:
        # model v0: self-trained
    ver = 'v0.2'

    log_dir = './_logs/training'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
            filename = os.path.join(log_dir, f'{ver}.log'),
            filemode = 'w',
            format='%(message)s',
            level=logging.INFO
    )
    logging.info('Game_#/Total_Games:winner,result,timestep_score,reason,model_epsilon_value')

    if ver == 'v0.0':
        train(32, 1000, ver) 
    elif ver == 'v0.1':
        train(16, 5000, ver) 
    elif ver == 'v0.2':
        train(8, 10000, ver)




