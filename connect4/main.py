import sys

from board import *
from rules import *
from model import *
from random_model import *
from simulation import *
from trainer import *


if __name__ == "__main__":
    rules = Rules(4)
    board = Board(6, 7)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "debugSim":
        model = RandomModel(board.width)
        simulation = Simulation(rules, board, model)
        simulation.run()
        simulation.print()

    if cmd == "train":
        model = Model(board.numCells(), board.width, board.width + 4, 0)
        trainer = Trainer(rules, board, model)
        trainer.train()
        trainer.saveModel()