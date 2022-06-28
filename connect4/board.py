import numpy as np

from parameters import *
from rules import *

class Board:
    def __init__(self, width, height):
        self.cells = None
        self.occupiedCells = 0
        self.width = width
        self.height = height
        self.heights = None
        self.reset()

    def reset(self):
        self.cells = np.full((self.height, self.width), Rules.ColorNone, dtype=np.int)
        self.heights = np.zeros(self.width, dtype=np.int)
        self.occupiedCells = 0

    def numCells(self):
        return self.width * self.height

    def isFull(self):
        return self.occupiedCells >= self.numCells()

    def clone(self):
        other = Board(self.width, self.height)
        other.cells = np.copy(self.cells)
        other.heights = np.copy(self.heights)
        other.occupiedCells = self.occupiedCells

    def algorithmState(self, color):
        empty_positions = np.where(self.cells == Rules.ColorNone, 1, 0)
        player_chips = np.where(self.cells == color, 1, 0)
        opponent_chips = np.where(self.cells == -color, 1, 0)
        return np.array([empty_positions, player_chips, opponent_chips])

    def __str__(self):
        row = "| "
        for c in range(self.width):
            row += f"{c} | "
        rows = f"{row}\n"

        row = "-"
        for c in range(self.width):
            row += "----"
        rows += f"{row}\n"

        for r in range(self.height - 1, -1, -1):
            row = "| "
            for c in range(self.width):
                color = self.cells[r, c]
                colorName = Rules.colorName(color)[0]
                row += f"{colorName} | "
            rows += f"{row}\n"
        return rows
