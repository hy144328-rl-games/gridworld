#!/usr/bin/env python3

"""Reinforcement learning of Gridworld."""

import abc
import enum
import random
import typing

import numpy as np

class Action(enum.Enum):
    """Action in Markov decision process."""
    NORTH = enum.auto()
    SOUTH = enum.auto()
    EAST = enum.auto()
    WEST = enum.auto()

class State(typing.NamedTuple):
    """State in Markov decision process."""
    row: int
    col: int

    def __add__(self, a: Action) -> "State":
        row = self.row
        col = self.col

        if a == Action.NORTH:
            row -= 1
        elif a == Action.SOUTH:
            row += 1
        elif a == Action.WEST:
            col -= 1
        elif a == Action.EAST:
            col += 1

        return State(row, col)

class Grid:
    """Gridworld lattice."""
    def __init__(self, no_rows: int=5, no_cols: int=5):
        self.no_rows: int = no_rows
        self.no_cols: int = no_cols

    @staticmethod
    def on_border_top(s: State) -> bool:
        """Checks whether on top border."""
        return s.row == 0

    def on_border_bottom(self, s: State) -> bool:
        """Checks whether on bottom border."""
        return s.row == self.no_rows - 1

    @staticmethod
    def on_border_left(s: State) -> bool:
        """Checks whether on left border."""
        return s.col == 0

    def on_border_right(self, s: State) -> bool:
        """Checks whether on right border."""
        return s.col == self.no_cols - 1

    def on_border(self, s: State) -> bool:
        """Checks whether on any border."""
        res = False

        res |= self.on_border_top(s)
        res |= self.on_border_bottom(s)
        res |= self.on_border_left(s)
        res |= self.on_border_right(s)

        return res

    def off_border_top(self, s: State, a: Action) -> bool:
        """Checks whether off top border."""
        return self.on_border_top(s) and a == Action.NORTH

    def off_border_bottom(self, s: State, a: Action) -> bool:
        """Checks whether off bottom border."""
        return self.on_border_bottom(s) and a == Action.SOUTH

    def off_border_left(self, s: State, a: Action) -> bool:
        """Checks whether off left border."""
        return self.on_border_left(s) and a == Action.WEST

    def off_border_right(self, s: State, a: Action) -> bool:
        """Checks whether off right border."""
        return self.on_border_right(s) and a == Action.EAST

    def off_border(self, s: State, a: Action) -> bool:
        """Checks whether off any border."""
        res = False

        res |= self.off_border_top(s, a)
        res |= self.off_border_bottom(s, a)
        res |= self.off_border_left(s, a)
        res |= self.off_border_right(s, a)

        return res

    def flatten(self, s: State) -> int:
        """Returns index in a flattened, one-dimensional array."""
        return s.row * self.no_cols + s.col

    def unflatten(self, idx: int) -> State:
        """Returns index in a two-dimensional array."""
        return State(idx // self.no_cols, idx % self.no_cols)

class Game(abc.ABC):
    """Environment in Markov decision process."""
    def __init__(self):
        self.gamma: float = 0.9
        self.no_iterations: int = 50
        self.no_samples: int = 1

    @abc.abstractmethod
    def reward(self, g: Grid, s: State, a: Action) -> float:
        """Calculates reward."""
        ...

    @abc.abstractmethod
    def policy(self, g: Grid, s: State, a: Action) -> float:
        """Calculates probability of action."""
        ...

    @abc.abstractmethod
    def transition(self, g: Grid, s: State, a: Action) -> float:
        """Calculates state."""
        ...

    def play(
        self,
        grid: Grid,
        initial_state: State,
    ):
        """Plays one or multiple games."""
        res = []

        for sample_ct in range(self.no_samples):
            random.seed(sample_ct)

            current_state = initial_state
            discount = 1.0
            val = 0.0

            for _ in range(self.no_iterations):
                weights = [self.policy(grid, current_state, a_it) for a_it in Action]
                next_action = random.choices(list(Action), weights)[0]
                val += discount * self.reward(grid, current_state, next_action)

                current_state = self.transition(grid, current_state, next_action)
                discount *= self.gamma

            res.append(val)

        return np.mean(res)

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

class SpecialCaseGame(Game):
    """Game with special cases for rewards and transitions."""
    def __init__(
        self,
        reward_special_cases: list[RewardSpecialCase] = None,
        transition_special_cases: list[TransitionSpecialCase] = None,
    ):
        super().__init__()

        self.reward_special_cases: list[RewardSpecialCase] = reward_special_cases or []
        self.transition_special_cases: list[TransitionSpecialCase] = transition_special_cases or []

    def reward(self, g: Grid, s: State, a: Action) -> float:
        for special_case_it in self.reward_special_cases:
            if s == special_case_it.state and a == special_case_it.action:
                return special_case_it.reward

        if g.off_border(s, a):
            return -1

        return 0

    def policy(self, g: Grid, s: State, a: Action) -> float:
        return 1 / len(Action)

    def transition(self, g: Grid, s: State, a: Action) -> float:
        for special_case_it in self.transition_special_cases:
            if s == special_case_it.state and a == special_case_it.action:
                return special_case_it.next_state

        if g.off_border(s, a):
            return s

        return s + a

def main_brute_force(
    grid,
    game,
):
    """Runs Monte-Carlo simulation."""
    res = np.empty((grid.no_rows, grid.no_cols))

    for i in range(grid.no_rows):
        for j in range(grid.no_cols):
            res[i, j] = game.play(
                grid,
                State(i, j),
            )
            print(i, j, res[i, j])

    print(res)

def main_hamilton_jacobi(grid, game):
    """Solves Hamilton-Jacobi equations."""
    no_cells = grid.no_rows * grid.no_cols
    A = np.zeros((no_cells, no_cells))
    b = np.zeros(no_cells)

    for i in range(grid.no_rows):
        for j in range(grid.no_cols):
            state = State(i, j)
            idx = grid.flatten(state)
            A[idx, idx] = -1

            for a_it in Action:
                pi = game.policy(grid, state, a_it)

                r = game.reward(grid, state, a_it)
                b[idx] -= pi * r

                new_state = game.transition(grid, state, a_it)
                new_idx = grid.flatten(new_state)
                A[idx, new_idx] += pi * game.gamma

    res = np.linalg.solve(A, b)
    res = res.reshape((grid.no_rows, grid.no_cols))
    print(res)

def main():
    """Main function."""
    A: State = State(0, 1)
    A_prime: State = State(4, 1)
    B: State = State(0, 3)
    B_prime: State = State(2, 3)

    grid = Grid(5, 5)
    game = SpecialCaseGame(
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

    game.gamma = 0.9
    game.no_iterations = 100
    game.no_samples = 1000

    main_brute_force(grid, game)
    main_hamilton_jacobi(grid, game)

if __name__ == "__main__":
    main()
