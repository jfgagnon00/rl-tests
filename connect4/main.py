import json
import os.path
import sys

from board import *
from rules import *
from model import *
from random_model import *
from simulation import *
from trainer import *


if __name__ == "__main__":
    filename = "connect4_model.bin"

    rules = Rules(4)
    board = Board(6, 7)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "debugLog":
        model = RandomModel(board.width)
        simulation = Simulation(rules, board, model)
        simulation.run()
        simulation.debugLog()

    elif cmd == "debugReturns":
        with open(f"{filename}.json", 'r') as f:
            modelParams = json.load(f)

        model = Model(**modelParams)
        model.load(filename)
        trainer = Trainer(rules, board, model)
        trainer.debugReturns()

    elif cmd == "train":
        modelParams = {
            'numInputs': board.numCells(),
            'numOutputs': board.width,
            'hiddenLayersNumFeatures': board.width + 4,
            'numHiddenLayers': 0,
        }

        with open(f"{filename}.json", 'w') as f:
            json.dump(modelParams, f)

        model = Model(**modelParams)

        if os.path.exists(filename):
            print(f"Loading weights from {filename}")
            model.load(filename)

        trainer = Trainer(rules, board, model)
        trainer.train()
        model.save(filename)

    elif cmd == "play":
        with open(f"{filename}.json", 'r') as f:
            modelParams = json.load(f)

        model = Model(**modelParams)
        model.load(filename)
        simulation = Simulation(rules, board, model)
        simulation.play()

    else:
        print("Unknown cmd")