#!/usr/bin/env python3

"""Agent in Markov decision process."""

import random

import numpy as np

from environment import Environment
from state import Action, State

class Agent:
    """Agent in Markov decision process."""
    def __init__(self, env: Environment, s: State):
        self.env: Environment = env
        self.initial_state: State = s
        self.current_state: State = self.initial_state

    def policy(self, a: Action) -> float:
        # pylint: disable=unused-argument,no-self-use
        """Calculates probability of action."""
        return 1 / len(Action)

    def pick(self) -> Action:
        """Picks action according to policy."""
        weights = [self.policy(a_it) for a_it in Action]
        next_action = random.choices(list(Action), weights)[0]
        return next_action

    def move(self) -> float:
        """Moves agent according to policy."""
        next_action = self.pick()
        res = self.env.reward(self.current_state, next_action)
        self.current_state = self.env.transition(self.current_state, next_action)

        return res

    def play(
        self,
        no_iterations: int = 50,
        no_samples: int = 1,
    ) -> float:
        """Plays one or multiple games."""
        res = []

        for sample_ct in range(no_samples):
            random.seed(sample_ct)

            self.current_state = self.initial_state
            discount = 1.0
            val = 0.0

            for _ in range(no_iterations):
                reward = self.move()
                val += discount * reward
                discount *= self.env.gamma

            res.append(val)

        return np.mean(res)
