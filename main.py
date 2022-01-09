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
    def __init__(self, no_rows: int=5, no_cols: int=5):
        self.no_rows: int = no_rows
        self.no_cols: int = no_cols

    def on_border_top(self, s: State) -> bool:
        return s.row == 0

    def on_border_bottom(self, s: State) -> bool:
        return s.row == self.no_rows - 1

    def on_border_left(self, s: State) -> bool:
        return s.col == 0

    def on_border_right(self, s: State) -> bool:
        return s.col == self.no_cols - 1

    def on_border(self, s: State) -> bool:
        res = False

        res |= self.on_border_top(s)
        res |= self.on_border_bottom(s)
        res |= self.on_border_left(s)
        res |= self.on_border_right(s)

        return res

    def off_border_top(self, s: State, a: Action) -> bool:
        return self.on_border_top(s) and a == Action.NORTH

    def off_border_bottom(self, s: State, a: Action) -> bool:
        return self.on_border_bottom(s) and a == Action.SOUTH

    def off_border_left(self, s: State, a: Action) -> bool:
        return self.on_border_left(s) and a == Action.WEST

    def off_border_right(self, s: State, a: Action) -> bool:
        return self.on_border_right(s) and a == Action.EAST

    def off_border(self, s: State, a: Action) -> bool:
        res = False

        res |= self.off_border_top(s, a)
        res |= self.off_border_bottom(s, a)
        res |= self.off_border_left(s, a)
        res |= self.off_border_right(s, a)

        return res

g: Grid = Grid(5, 5)
A: State = State(0, 1)
A_prime: State = State(4, 1)
B: State = State(0, 3)
B_prime: State = State(2, 3)

def reward(g: Grid, s: State, a: Action) -> float:
    if s == A:
        return 10
    elif s == B:
        return 5
    elif g.off_border(s, a):
        return -1
    else:
        return 0

def policy(g: Grid, s: State, a: Action) -> float:
    return 0.25

def transition(g: Grid, s: State, a: Action) -> State:
    if s == A:
        return A_prime
    elif s == B:
        return B_prime
    elif g.off_border(s, a):
        return s
    else:
        return s + a

if __name__ == "__main__":
    import random

    s = State(0, 0)
    gamma = 0.9
    val = 0.0

    for i in range(50):
        weights = [policy(g, s, a_it) for a_it in Action]
        a = random.choices([a_it for a_it in Action], weights)[0]
        r = reward(g, s, a)
        val += gamma**i * r
        print(s, a, r, val)

        s = transition(g, s, a)

    print(val)

