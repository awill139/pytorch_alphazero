import numpy as np
import torch
import torch.nn as nn
import time

from utils import *
from alpha_model import Model
from mcts import MCTS
from gen_env import make_game
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_units):
    episode_returns = []
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    D = Database(max_size=data_size,batch_size=batch_size)        
    model = Model(Env=Env, n_hidden_units=n_hidden_units)  
    model = model.to(device)
    t_total = 0   
    R_best = -np.Inf

    for ep in range(n_ep):    
        start = time.time()
        s = Env.reset() 
        R = 0.0
        a_store = []
        seed = np.random.seed(13)
        Env.seed(seed)      
        if is_atari: 
            mcts_env.reset()
            mcts_env.seed(seed)                                

        mcts = MCTS(root_index=s, root=None, model=model, na=model.action_dim, gamma=gamma)                          
        for t in range(max_ep_len):
            mcts.search(n_mcts = n_mcts, c = c,Env = Env, mcts_env = mcts_env)
            state, pi, V = mcts.return_results(temp)
            D.store((state,V,pi))

            a = np.random.choice(len(pi),p=pi)
            a_store.append(a)
            s1, r, done, _ = Env.step(a)
            R += r
            t_total += n_mcts                

            if done:
                break
            else:
                mcts.forward(a,s1)
        
        episode_returns.append(R)
        timepoints.append(t_total)
        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        print('epi {}, tot ret: {}, tot time: {}'.format(ep, np.round(R, 2), np.round((time.time()-start))))




        crit_v = nn.MSELoss()
        crit_p = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        D.reshuffle()
        for epoch in range(1):
            for sb, Vb, pib in D:
                optimizer.zero_grad()
                val, pol = model(torch.tensor(sb, dtype = torch.float).to(device = device))
                val_loss = crit_v(val, torch.tensor(Vb.tolist(), dtype = torch.float).to(device = device))
                #print(pol.shape)
                #pib = np.squeeze(pib)
                pib = pib.T
                pol_loss = crit_p(pol, torch.tensor(pib[0].tolist(), dtype = torch.long).to(device = device))
                loss = val_loss + pol_loss
                loss.backward()
                optimizer.step()

    return episode_returns, timepoints, a_best, seed_best, R_best
