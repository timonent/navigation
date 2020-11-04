import os
import numpy as np
import torch
import torch.nn as nn

from config import hyperparameters as h
from config import model_dir

#----------------------------------------------------------------------------
# Deep Q-network.

class Q(nn.Module):
    def __init__(self, state_shape, nof_actions, network_role, name='dueling_dqn'):
        super(Q, self).__init__()
        self.state_shape = state_shape
        self.in_channels = state_shape[0]
        self.network_role = network_role

        fs1, fs2, fs3 = h.nof_filters

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.in_channels, fs1, (1, 3, 3), stride=(1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(fs1, fs2, (1, 3, 3), stride=(1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(fs2, fs3, (h.frame_stack_size, 3, 3), stride=(1, 3, 3)),
            nn.ReLU()
        )
        self.conv_block_output_size = self.conv_block_output_size()
        self.fc_common = nn.Sequential(
            nn.Linear(self.conv_block_output_size, h.fc_size),
            nn.ReLU(),
            nn.Linear(h.fc_size, h.fc_size),
            nn.ReLU()
        )
        self.state_value_stream = nn.Linear(h.fc_size, 1)
        self.advantage_stream = nn.Linear(h.fc_size, nof_actions)

        self.checkpoint_file = os.path.join(model_dir, f'{name}_{network_role}')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=h.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = self.conv_block(state).reshape(-1, self.conv_block_output_size)
        x = self.fc_common(x)
        val = self.state_value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(1, keepdim=True))
            
    def conv_block_output_size(self):
        dummy = torch.zeros(1, *self.state_shape)
        dims = self.conv_block(dummy)
        return np.prod(dims.shape)

    def save_checkpoint(self):
        print(f'Saving checkpoint for the {self.network_role} network...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f'Loading checkpoint for the {self.network_role} network...')
        self.load_state_dict(torch.load(self.checkpoint_file))

#----------------------------------------------------------------------------
