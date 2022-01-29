#!/usr/bin/env python3

"""Reinforcement learning of Gridworld."""

import typing

import numpy as np

from agent import Agent
from environment import Environment, Grid
from state import Action, State

class RewardSpecialCase(typing.NamedTuple):
    """Special case for reward."""
    state: State
    action: Action
    reward: float

class TransitionSpecialCase(typing.NamedTuple):
    """Special case for transition."""
    state: State
    action: Action
    next_state: State

class SpecialCaseEnvironment(Environment):
    """Environment with special cases for rewards and transitions."""
    def __init__(
        self,
        grid: Grid,
        reward_special_cases: typing.List[RewardSpecialCase] = None,
        transition_special_cases: typing.List[TransitionSpecialCase] = None,
    ):
        super().__init__(grid)

        self.reward_special_cases: typing.List[RewardSpecialCase] = \
            reward_special_cases or []
        self.transition_special_cases: typing.List[TransitionSpecialCase] = \
            transition_special_cases or []

    def reward(self, s: State, a: Action) -> float:
        for special_case_it in self.reward_special_cases:
            if s == special_case_it.state and a == special_case_it.action:
                return special_case_it.reward

        return super().reward(s, a)

    def transition(self, s: State, a: Action) -> float:
        for special_case_it in self.transition_special_cases:
            if s == special_case_it.state and a == special_case_it.action:
                return special_case_it.next_state

        return super().transition(s, a)

def main_monte_carlo(env: Environment):
    """Runs Monte-Carlo simulation."""
    res = np.empty((env.grid.no_rows, env.grid.no_cols))

    for i in range(env.grid.no_rows):
        for j in range(env.grid.no_cols):
            agent = Agent(
                env,
                State(i, j),
            )
            res[i, j] = agent.play(
                no_iterations = 100,
                no_samples = 1000,
            )
            print(i, j, res[i, j])

    return res

def main_hamilton_jacobi(env: Environment):
    """Solves Hamilton-Jacobi equations."""
    no_cells = env.grid.no_rows * env.grid.no_cols
    A = np.zeros((no_cells, no_cells))
    b = np.zeros(no_cells)

    for i in range(env.grid.no_rows):
        for j in range(env.grid.no_cols):
            state = State(i, j)
            agent = Agent(env, state)

            idx = env.grid.flatten(state)
            A[idx, idx] = -1

            for a_it in Action:
                pi = agent.policy(a_it)

                r = env.reward(state, a_it)
                b[idx] -= pi * r

                new_state = env.transition(state, a_it)
                new_idx = env.grid.flatten(new_state)
                A[idx, new_idx] += pi * env.gamma

    res = np.linalg.solve(A, b)
    return res.reshape((env.grid.no_rows, env.grid.no_cols))

def main():
    """Main function."""
    grid = Grid(5, 5)
    env = special_case_environment(grid)

    print(main_monte_carlo(env))
    print(main_hamilton_jacobi(env))

if __name__ == "__main__":
    main()
