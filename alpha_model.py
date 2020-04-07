import torch
from utils import check_space
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    
    def __init__(self, Env, n_hidden_units):
        super(Model, self).__init__()
        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)


        self.fc = nn.Linear(self.state_dim[0], n_hidden_units)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.elu2 = nn.ELU()
        self.pol = nn.Linear(n_hidden_units, self.action_dim)
        self.soft = nn.LogSoftmax()
        self.val = nn.Linear(n_hidden_units, 1)

    def forward(self, sb):
        x = self.elu(self.fc(sb))
        x = self.elu2(self.fc2(x))
        v = self.val(x)
        p = self.pol(x)
        
        # print(p.shape)
        # print(p.view(p.size(0)))
        p = self.soft(p)

        return v, p
