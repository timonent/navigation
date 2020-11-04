import numpy as np

from config import hyperparameters as h

#----------------------------------------------------------------------------
# Experience replay with a uniform sampling scheme.

class ExperienceReplay:
    def __init__(self, state_shape):
        self.state_shape = state_shape

        self.store_counter = 0
        self.states = np.empty((h.replay_mem_size, *state_shape), dtype=np.float32)
        self.actions = np.empty(h.replay_mem_size, dtype=np.int64)
        self.rewards = np.empty(h.replay_mem_size, dtype=np.float32)
        self.states_prime = np.empty((h.replay_mem_size, *state_shape), dtype=np.float32)
        self.terminals = np.empty(h.replay_mem_size, dtype=np.uint8)

    def store(self, s, a, r, s_prime, done):
        idx = self.store_counter % h.replay_mem_size
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.states_prime[idx] = s_prime
        self.terminals[idx] = done
        self.store_counter += 1

    def sample(self):
        population_size = min(self.store_counter, h.replay_mem_size)
        indices = np.random.choice(population_size, size=h.batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_prime = self.states_prime[indices]
        terminals = self.terminals[indices]

        return (states, actions, rewards, states_prime, terminals)

#----------------------------------------------------------------------------
