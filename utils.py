import os
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
plt.rcParams.update({
    # LaTeX
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    # axes
    'axes.spines.top': False,
    'axes.spines.right': False
})
plt.style.use('fivethirtyeight')

from config import hyperparameters as h
from config import plot_dir

#----------------------------------------------------------------------------
# Wrapper for UnityEnvironment enabling frame stacking.

class UnityEnvironmentWrapper:
    def __init__(self, env, brain_name, train_mode=False):
        self.env = env
        self.brain_name = brain_name
        self.train_mode = train_mode

        self.frame_stack = deque(maxlen=h.frame_stack_size)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        observation = env_info.visual_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        self.frame_stack.appendleft(observation)

        return (self.frame_stack_reshaped(), reward, done)

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        for _ in range(h.frame_stack_size - 1):
            self.frame_stack.appendleft(np.zeros(env_info.visual_observations[0].shape))
        
        self.frame_stack.appendleft(env_info.visual_observations[0])

        return self.frame_stack_reshaped()

    def close(self):
        self.env.close()

    def frame_stack_reshaped(self):
        state = np.asarray(self.frame_stack)

        return np.squeeze(np.transpose(state, (1, 4, 0, 2, 3)), axis=0) # (3, h.frame_stack_size, 84, 84)

#----------------------------------------------------------------------------
# Plotting.

def plot(training_steps, scores):
    try:
        with open(f'{os.path.join(plot_dir, "training_steps.npy")}', 'wb') as f:
            np.save(f, np.asarray(training_steps))
        with open(f'{os.path.join(plot_dir, "scores.npy")}', 'wb') as f:
            np.save(f, np.asarray(scores))
    except:
        print('Error while backing up training data, continuing without.')

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot()

    ax.plot(training_steps, scores, color='mediumslateblue', linewidth=1.5, alpha=0.4)

    n = len(scores)
    avg_scores = np.empty(n)
    window_size = 100
    for i in range(n):
        avg_scores[i] = np.mean(scores[max(0, i-window_size):(i+1)])

    ax.plot(training_steps, avg_scores, color='indigo', linewidth=5.0)
    ax.set_xlabel(r'$\mathrm{Training \ steps \ (in \ thousands)}$')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fr'${int(x/1000)}$'))
    ax.set_ylabel(r'$\mathrm{Score}$')

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    file_name = 'scores_plot.svg'
    plt.savefig(os.path.join(plot_dir, file_name), format='svg')

    print(f'Training progress plot saved to {plot_dir}/{file_name}')
    
#----------------------------------------------------------------------------
