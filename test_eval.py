#!/usr/bin/env python3

"""Tests policy evaluation."""

import abc
import typing

import numpy as np
import pytest

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

class PolicyEvaluationTest(abc.ABC):
    # pylint: disable=no-self-use
    """Tests policy evaluation."""
    @pytest.fixture
    def grid(self) -> Grid:
        """Returns grid."""
        return Grid(5, 5)

    @pytest.fixture
    def A(self) -> State:
        """First special case."""
        return State(0, 1)

    @pytest.fixture
    def B(self) -> State:
        """Second special case."""
        return State(0, 3)

    @pytest.fixture
    def reward_special_cases(self, A: State, B: State) -> typing.List[RewardSpecialCase]:
        """Rewards for special cases."""
        return [
            RewardSpecialCase(A, action_it, 10)
            for action_it in Action
        ] + [
            RewardSpecialCase(B, action_it, 5)
            for action_it in Action
        ]

    @pytest.fixture
    def transition_special_cases(self, A: State, B: State) -> typing.List[TransitionSpecialCase]:
        """Transitions for special cases."""
        return [
            TransitionSpecialCase(A, action_it, State(4, 1))
            for action_it in Action
        ] + [
            TransitionSpecialCase(B, action_it, State(2, 3))
            for action_it in Action
        ]

    @pytest.fixture
    def env(
        self,
        grid: Grid,
        reward_special_cases: typing.List[RewardSpecialCase],
        transition_special_cases: typing.List[TransitionSpecialCase],
    ) -> Environment:
        """Builds environment."""
        return SpecialCaseEnvironment(
            grid,
            reward_special_cases = reward_special_cases,
            transition_special_cases = transition_special_cases,
        )

    @abc.abstractmethod
    def value_function(self, env: Environment) -> np.ndarray:
        """Computes value function."""

    def test(self, value_function: np.ndarray, tol: float=5E-2):
        """Compares to reference solution."""
        assert value_function == pytest.approx(
            np.array([
                [ 3.3,  8.8,  4.4,  5.3,  1.5],
                [ 1.5,  3.0,  2.3,  1.9,  0.5],
                [ 0.1,  0.7,  0.7,  0.4, -0.4],
                [-1.0, -0.4, -0.4, -0.6, -1.2],
                [-1.9, -1.3, -1.2, -1.4, -2.0],
            ]),
            abs = tol,
        )

class TestHamiltonJacobi(PolicyEvaluationTest):
    # pylint: disable=no-self-use
    """Tests Hamilton-Jacobi equations."""
    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Solves Hamilton-Jacobi equations."""
        grid = env.grid
        no_cells = grid.no_rows * grid.no_cols

        A = np.zeros((no_cells, no_cells))
        b = np.zeros(no_cells)

        for i in range(grid.no_rows):
            for j in range(grid.no_cols):
                state = State(i, j)
                agent = Agent(env, state)

                idx = grid.flatten(state)
                A[idx, idx] = -1

                for a_it in Action:
                    pi = agent.policy(state, a_it)
                    r = env.reward(state, a_it)
                    b[idx] -= pi * r

                    new_state = env.transition(state, a_it)
                    new_idx = grid.flatten(new_state)
                    A[idx, new_idx] += pi * env.gamma

        res = np.linalg.solve(A, b)
        return res.reshape((grid.no_rows, grid.no_cols))

class TestMonteCarlo(PolicyEvaluationTest):
    # pylint: disable=no-self-use
    """Tests Monte-Carlo simulations."""
    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Performs Monte-Carlo simulations."""
        grid = env.grid
        res = np.empty((grid.no_rows, grid.no_cols))

        for i in range(grid.no_rows):
            for j in range(grid.no_cols):
                agent = Agent(
                    env,
                    State(i, j),
                )
                res[i, j] = agent.play(
                    no_iterations = 100,
                    no_samples = 1000,
                )

        return res

    def test(self, value_function: np.ndarray):
        super().test(value_function, tol=2E-1)
