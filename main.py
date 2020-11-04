import sys
import argparse
import time
import numpy as np
from unityagents import UnityEnvironment

from config import hyperparameters as h, env_file
from utils import UnityEnvironmentWrapper

from agent import Agent

#----------------------------------------------------------------------------
# Training.

def train(args):
    env = UnityEnvironment(file_name=env_file)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    state_shape = env_info.visual_observations[0].shape
    nof_actions = brain.vector_action_space_size

    env = UnityEnvironmentWrapper(env, brain_name, train_mode=True)
    agent = Agent((state_shape[3], h.frame_stack_size, state_shape[1], state_shape[2]), nof_actions)

    training_steps, scores = [], []
    step_counter = 0
    for episode in range(1, h.episodes+1):
        state = env.reset()
        score = 0
        for t in range(h.t_max):
            action = agent.act(state)
            observation, reward, done = env.step(action)
            agent.step(state, action, reward, observation, done)
            score += reward
            state = observation
            step_counter += 1
            if done:
                break
        
        training_steps.append(step_counter)
        scores.append(score)

        if episode % 100 == 0:
            print(f'On episode {episode}, average score {np.mean(scores[-100:])}, epsilon {agent.epsilon:.2f}')
            agent.save_models()

    if args.plot:
        from utils import plot
        plot(training_steps, scores)

    env.close()

#----------------------------------------------------------------------------
# Testing.

def test(args):
    env = UnityEnvironment(file_name=env_file, seed=1111)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=False)[brain_name]
    state_shape = env_info.visual_observations[0].shape
    nof_actions = brain.vector_action_space_size

    env = UnityEnvironmentWrapper(env, brain_name, train_mode=False)
    agent = Agent((state_shape[3], h.frame_stack_size, state_shape[1], state_shape[2]), nof_actions)
    agent.load_models()
    agent.epsilon = 0.0

    for episode in range(1, args.episodes+1):
        state = env.reset()
        score = 0
        for t in range(h.t_max):
            action = agent.act(state)
            observation, reward, done = env.step(action)
            score += reward
            state = observation
            if done:
                break

        print(f'Score on testing episode {episode}: {score}')

    env.close()

#----------------------------------------------------------------------------
# Entry point.

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        description = 'Tool for training and running the reinforcement learning agent \
                       in the Visual Banana environment.',
        epilog      = f'Use "{prog} <command> -h" for more information.')

    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser('test', help='Watch the trained agent conquer the environment')
    p.add_argument('-e', '--episodes', dest='episodes', default=1, type=int)
    p.set_defaults(func=test)

    p = subparsers.add_parser('train', help='Train the agent')
    p.add_argument('-p', '--plot', dest='plot', action='store_true')
    p.set_defaults(func=train)
    
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    args.func(args)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
