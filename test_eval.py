#!/usr/bin/env python3

"""Tests Monte-Carlo simulations and Hamilton-Jacobi equations."""

import numpy as np
import pytest

from environment import Environment, Grid
from main import main_hamilton_jacobi, special_case_environment

class TestHamiltonJacobi:
    """Tests Hamilton-Jacobi equations."""
    @pytest.fixture
    def grid(self) -> Grid:
        """Returns grid."""
        return Grid(5, 5)

    @pytest.fixture
    def env(self, grid: Grid):
        """Returns environment."""
        return special_case_environment(grid)

    @pytest.fixture
    def value_function(self, env: Environment) -> np.ndarray:
        """Value grid."""
        return main_hamilton_jacobi(env)

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
