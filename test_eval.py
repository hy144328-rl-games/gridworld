#!/usr/bin/env python3

"""Tests policy evaluation."""

import abc
import typing

import numpy as np
import pytest

from environment import Environment, Grid
from main import main_hamilton_jacobi, main_monte_carlo
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
    """Tests policy evaluation."""
    @pytest.fixture
    def grid(self) -> Grid:
        """Returns grid."""
        return Grid(5, 5)

    @pytest.fixture
    def env(self, grid: Grid) -> Environment:
        """Returns environment."""
        A: State = State(0, 1)
        A_prime: State = State(4, 1)
        B: State = State(0, 3)
        B_prime: State = State(2, 3)

        env = SpecialCaseEnvironment(
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

        return env

    @abc.abstractmethod
    def value_function(self, env: Environment) -> np.ndarray:
        ...

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
    """Tests Hamilton-Jacobi equations."""
    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Solves Hamilton-Jacobi equations."""
        return main_hamilton_jacobi(env)

class TestMonteCarlo(PolicyEvaluationTest):
    """Tests Monte-Carlo simulations."""
    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Performs Monte-Carlo simulations."""
        return main_monte_carlo(env)

    def test(self, value_function: np.ndarray):
        super().test(value_function, tol=2E-1)
