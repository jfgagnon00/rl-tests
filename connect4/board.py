import numpy as np
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
        self.cells = np.full((self.height, self.width), Rules.ColorNone, dtype=np.float)
        self.heights = np.zeros(self.width, dtype=np.int)
        self.occupiedCells = 0

    def numCells(self):
        return self.width * self.height

    def isFull(self):
        return self.occupiedCells >= self.numCells()

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
