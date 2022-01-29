#!/usr/bin/env python3

"""States and actions in Markov decision process."""

import enum
import typing

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
