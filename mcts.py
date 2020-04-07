import numpy as np
import torch
import copy
from utils import is_atari_game, stable_normalizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class State():

    def __init__(self, index, r, done, parent_action, na, model):

        self.index = index
        self.r = r # reward upon arriving in this state
        self.done = done # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model#.to(device = device)
        
        self.evaluate()
        self.na = na
        self.child_actions = [Action(a, parent_state=self, Q_init = self.V) for a in range(na)]
        _, self.priors = model(torch.tensor(index[None,], dtype = torch.float).to(device = device))
        self.priors = self.priors.flatten()
    
    def select(self,c=1.5):
        UCT = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action, prior in zip(self.child_actions,self.priors)]) 
        winner = np.argmax(UCT)
        return self.child_actions[winner]

    def evaluate(self):
        if not self.done:
            self.V, _ = self.model(torch.tensor(self.index[None, ], dtype = torch.float).to(device = device))
            self.V = np.squeeze(self.V)
        else:
            self.V = np.array(0.0)  

    def update(self):
        self.n += 1

class Action():
    def __init__(self,index,parent_state,Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
                
    def add_child_state(self, s1, r, done, model):
        self.child_state = State(s1, r, done, self, self.parent_state.na, model)
        return self.child_state
        
    def update(self,R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n

class MCTS():
    def __init__(self, root, root_index, model, na, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
    
    def search(self, n_mcts, c, Env, mcts_env):
        if self.root is None:
            self.root = State(self.root_index, r=0.0, done=False, parent_action=None, na=self.na, model=self.model)
        else:
            self.root.parent_action = None

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)   
        
        for i in range(n_mcts):     
            state = self.root
            if not is_atari:
                mcts_env = copy.deepcopy(Env) 
            else:
                restore_atari_state(mcts_env, snapshot)            
            
            while not state.done: 
                action = state.select(c = c)
                s1, r, d, _ = mcts_env.step(action.index)
                if hasattr(action,'child_state'):
                    state = action.child_state
                    continue
                else:
                    state = action.add_child_state(s1, r, d, self.model)
                    break

            R = state.V         
            while state.parent_action is not None: 
                R = state.r + self.gamma * R 
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self,temp):
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index, pi_target, V_target
    
    def forward(self,a,s1):
        if not hasattr(self.root.child_actions[a],'child_state'):
            self.root = None
            self.root_index = s1      
        else:
            self.root = self.root.child_actions[a].child_state
