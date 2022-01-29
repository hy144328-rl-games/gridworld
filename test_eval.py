#!/usr/bin/env python3

"""Tests Monte-Carlo simulations and Hamilton-Jacobi equations."""

import numpy as np
import pytest

from agent import Agent
from environment import Environment, Grid
from main import RewardSpecialCase, SpecialCaseEnvironment, TransitionSpecialCase
from state import Action, State

class TestHamiltonJacobi:
    """Tests Hamilton-Jacobi equations."""
    @pytest.fixture
    def grid(self) -> Grid:
        """Returns grid."""
        return Grid(5, 5)

    @pytest.fixture
    def env(self, grid: Grid):
        """Returns environment."""
        A: State = State(0, 1)
        A_prime: State = State(4, 1)
        B: State = State(0, 3)
        B_prime: State = State(2, 3)

        return SpecialCaseEnvironment(
            grid,
            reward_special_cases = [
                RewardSpecialCase(A, action_it, 10)
                for action_it in Action
            ] + [
                RewardSpecialCase(B, action_it, 5)
                for action_it in Action
            ],
            transition_special_cases = [
                TransitionSpecialCase(A, action_it, A_prime)
                for action_it in Action
            ] + [
                TransitionSpecialCase(B, action_it, B_prime)
                for action_it in Action
            ],
        )

    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Value grid."""
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

    def test(self, value_function: np.ndarray):
        """Compares to reference solution."""
        assert value_function == pytest.approx(
            np.array([
                [ 3.3,  8.8,  4.4,  5.3,  1.5],
                [ 1.5,  3.0,  2.3,  1.9,  0.5],
                [ 0.1,  0.7,  0.7,  0.4, -0.4],
                [-1.0, -0.4, -0.4, -0.6, -1.2],
                [-1.9, -1.3, -1.2, -1.4, -2.0],
            ]),
            abs = 5E-2,
        )

