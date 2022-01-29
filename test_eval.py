#!/usr/bin/env python3

"""Tests policy evaluation."""

import abc
import numpy as np
import pytest

from environment import Environment, Grid
from main import main_hamilton_jacobi, main_monte_carlo, special_case_environment

class PolicyEvaluationTest(abc.ABC):
    """Tests policy evaluation."""
    @pytest.fixture
    def grid(self) -> Grid:
        """Returns grid."""
        return Grid(5, 5)

    @pytest.fixture
    def env(self, grid: Grid):
        """Returns environment."""
        return special_case_environment(grid)

    @abc.abstractmethod
    def value_function(self, env: Environment) -> np.ndarray:
        """Computes value function."""
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
