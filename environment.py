"""

### NOTICE ###
You DO NOT need to upload this file

"""
from typing import SupportsFloat

import gymnasium as gym
import numpy as np


class Environment(object):
    def __init__(self, env, args, test=False):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.do_render = args.do_render

        if args.video_dir:
            self.env = gym.wrappers.Monitor(self.env, args.video_dir, force=True)

    def seed(self, seed):
        """
        Control the randomness of the environment
        """
        self.env.seed(seed)

    def reset(self):
        """
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current state of the game, shape: (8)
        """
        observation, info = self.env.reset()
        return np.array(observation), info

    def step(self, action):
        """
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?

        When running pg:
            observation: np.array
                current state of the game, shape: (8)
            reward: int
            done: bool
                whether reach the end of the episode?
        """
        if not self.env.action_space.contains(action):
            raise ValueError("Ivalid action!!")

        if self.do_render:
            self.env.render()

        observation, reward, terminated, truncated, info = self.env.step(action)

        return np.array(observation), reward, terminated, truncated, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()
