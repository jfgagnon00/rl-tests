import json
import os.path
import sys

from board import *
from rules import *
from conv_model import *
from simple_model import *
from random_model import *
from simulation import *
from trainer import *


def modelFilename(modelClass):
    return f"connect4_{modelClass.__name__}.bin"

def initModel(modelClass, modelParams=None):
    filename = modelFilename(modelClass)

    if modelParams is None:
        with open(f"{filename}.json", 'r') as f:
            modelParams = json.load(f)

    model = modelClass(**modelParams)
    if os.path.exists(filename):
        print(f"Loading weights from {filename}")
        model.load(filename)

    return model

def saveModel(model, modelParams):
    filename = modelFilename(type(model))

    with open(f"{filename}.json", 'w') as f:
        json.dump(modelParams, f)

    model.save(filename)

def saveRewards(totalRewardHistory):
    i = 0
    while True:
        filename = f"total-history{i}.json"

        if os.path.exists(filename):
            i += 1
            continue

        with open(filename, "w") as f:
            data = {
                "totalRewardHistory": totalRewardHistory
            }
            json.dump(data, f)

        break


if __name__ == "__main__":
    rules = Rules(4)
    board = Board(6, 7)

    if False:
        modelClass = SimpleModel
        modelParams = {
            'numInputs': board.numCells(),
            'numOutputs': board.width,
            'hiddenLayersNumFeatures': board.width + 4,
            'numHiddenLayers': 0,
        }
    else:
        modelClass = ConvModel
        modelParams = {
            'numInputs': board.numCells(),
            'numOutputs': board.width,
        }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "debugLog":
        model = RandomModel(board.width)
        simulation = Simulation(rules, board, model)
        simulation.run()
        simulation.debugLog()

    elif cmd == "debugReturns":
        model = initModel(modelClass)
        trainer = Trainer(rules, board, model)
        trainer.debugReturns()

    elif cmd == "train":
        model = initModel(modelClass, modelParams)
        trainer = Trainer(rules, board, model)
        trainer.train()
        saveModel(model, modelParams)
        saveRewards(trainer.totalRewardHistory)

    elif cmd == "play":
        model = initModel(modelClass)
        simulation = Simulation(rules, board, model)
        simulation.play()

    else:
        print("Unknown cmd")