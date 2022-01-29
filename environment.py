#!/usr/bin/env python3

"""Environment in Markov decision process."""

from state import Action, State

class Grid:
    """Gridworld lattice."""
    def __init__(self, no_rows: int=5, no_cols: int=5):
        self.no_rows: int = no_rows
        self.no_cols: int = no_cols

    def on_border_top(self, s: State) -> bool:
        """Checks whether on top border."""
        return s.row == 0

    def on_border_bottom(self, s: State) -> bool:
        """Checks whether on bottom border."""
        return s.row == self.no_rows - 1

    def on_border_left(self, s: State) -> bool:
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

class Environment:
    """Environment in Markov decision process."""
    def __init__(self, grid: Grid):
        self.grid: Grid = grid
        self.gamma: float = 0.9

    def reward(self, s: State, a: Action) -> float:
        """Calculates reward."""
        if self.grid.off_border(s, a):
            return -1

        return 0

    def transition(self, s: State, a: Action) -> float:
        """Calculates state."""
        if self.grid.off_border(s, a):
            return s

        return s + a
