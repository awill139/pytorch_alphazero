import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import matplotlib.pyplot as plt
import numpy as np
import os
from agt import agent
import torch
from utils import smooth, symmetric_remove
from gen_env import make_game
from gym import wrappers

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='CartPole-v0')
parser.add_argument('--n_ep', type=int, default=500)
parser.add_argument('--n_mcts', type=int, default=25)
parser.add_argument('--max_ep_len', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--c', type=float, default=1.5)
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--window', type=int, default=25)

parser.add_argument('--n_hidden_units', type=int, default=128)


if __name__ == '__main__':

    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device: {}'.format(device))

    episode_returns, timepoints, a_best, seed_best, R_best = agent(game = args.game, n_ep = args.n_ep, n_mcts = args.n_mcts,
                                        max_ep_len = args.max_ep_len, lr = args.lr, c = args.c, gamma = args.gamma,
                                        data_size = args.data_size, batch_size = args.batch_size, temp = args.temp,
                                        n_hidden_units = args.n_hidden_units)

    fig,ax = plt.subplots(1,figsize=[7,5])
    total_eps = len(episode_returns)
    print(episode_returns)
    #episode_returns = smooth(episode_returns, args.window, mode='valid') 
    # print(episode_returns)
    ax.plot(np.arange(total_eps), episode_returns, linewidth=4, color='blue')
    ax.set_ylabel('Ret')
    ax.set_xlabel('Epi', color='blue')
    plt.savefig(os.getcwd() + '/learning_curve.png', bbox_inches="tight", dpi=300)


    #running into an error with libav-tools
    # Env = make_game(args.game)
    # Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
    # Env.reset()
    # Env.seed(seed_best)
    # for a in a_best:
    #     Env.step(a)
    #     Env.render()
