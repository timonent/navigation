import numpy as np
import torch

from experience_replay import ExperienceReplay
from network import Q
from config import hyperparameters as h

#----------------------------------------------------------------------------
# Reinforcement learning agent.

class Agent:
    def __init__(self, state_shape, nof_actions):
        self.state_shape = state_shape
        self.nof_actions = nof_actions
        
        self.epsilon = h.epsilon
        self.replay_buffer = ExperienceReplay(state_shape)
        self.Q = Q(state_shape, nof_actions, 'online')
        self.Q_tgt = Q(state_shape, nof_actions, 'target')
        self.Q_tgt.load_state_dict(self.Q.state_dict())
        self.device = self.Q.device

    def act(self, state):
        if np.random.sample() < self.epsilon:
            return np.random.randint(self.nof_actions)
        else:
            state = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32, device=self.device)
            return torch.argmax(self.Q(state)).item()

    def step(self, state, action, reward, state_, done):
        self.replay_buffer.store(state, action, reward, state_, done)
        if self.replay_buffer.store_counter % h.update_interval == 0 \
           and self.replay_buffer.store_counter >= h.batch_size:
            self.learn()

    def learn(self):
        s, a, r, s_prime, done = self.sample_memory()

        Q_estimate_next = self.Q_tgt(s_prime).gather(1, torch.argmax(self.Q(s_prime), 1, keepdim=True)).squeeze(1)
        Q_target = r + (h.gamma * Q_estimate_next * (1 - done.to(torch.float32)))
        Q_estimate = self.Q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = self.Q.loss(Q_estimate, Q_target.detach()).to(self.device)
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()

        self.soft_update()
        self.decrement_epsilon()

    def sample_memory(self):
        sample = self.replay_buffer.sample()
        return tuple(map(lambda x: torch.tensor(x).to(self.device), sample))

    def soft_update(self):
        for theta, theta_tgt in zip(self.Q.parameters(), self.Q_tgt.parameters()):
            theta_tgt.data.copy_(h.tau*theta.data + (1 - h.tau)*theta_tgt.data)

    def decrement_epsilon(self):
        self.epsilon = max(h.epsilon_min, self.epsilon * h.epsilon_decay)

    def save_models(self):
        self.Q.save_checkpoint()
        self.Q_tgt.save_checkpoint()

    def load_models(self):
        self.Q.load_checkpoint()
        self.Q_tgt.load_checkpoint()          
        
#----------------------------------------------------------------------------
