#!/usr/bin/env python3

import enum
import typing

class Action(enum.Enum):
    NORTH = enum.auto()
    SOUTH = enum.auto()
    EAST = enum.auto()
    WEST = enum.auto()

class State(typing.NamedTuple):
    row: int
    col: int

if __name__ == "__main__":
    for action_it in Action:
        print(action_it)

    for i in range(5):
        for j in range(5):
            pass
