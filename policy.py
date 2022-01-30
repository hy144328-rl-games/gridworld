#!/usr/bin/env python3

"""Policies in Markov decision process."""

import typing

import numpy as np

from environment import Environment
from state import Action, State

class Policy:
    """Policies in Markov decision process."""
    def __init__(
        self,
        env: Environment,
    ):
        self.env: Environment = env

        self.value_function: np.ndarray = np.zeros(
            (self.env.grid.no_rows, self.env.grid.no_cols),
        )
        self.heuristic: typing.Callable[
            typing.Dict[Action, float],
            typing.Dict[Action, float],
        ] = lambda x: {action_it: 0.25 for action_it in Action}

    def __call__(
        self,
        s: State,
    ) -> typing.Dict[Action, float]:
        # pylint: disable=unused-argument,no-self-use
        """Calculates probability of action."""
        action_values = {}
        for action_it in Action:
            new_state = self.env.transition(s, action_it)
            action_values[action_it] = self.value_function[new_state]

        action_probabilities = self.heuristic(action_values)
        return action_probabilities
