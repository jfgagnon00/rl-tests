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

def initModel(modelClass, modelParams=None, torchDevice=None):
    filename = modelFilename(modelClass)

    if modelParams is None:
        with open(f"{filename}.json", 'r') as f:
            modelParams = json.load(f)
    modelParams["device"] = torchDevice

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

def saveRewards(expectedReturnHistory):
    i = 0
    while True:
        filename = f"expected-return-history{i}.json"

        if os.path.exists(filename):
            i += 1
            continue

        with open(filename, "w") as f:
            data = {
                "expectedReturnHistory": expectedReturnHistory
            }
            json.dump(data, f)

        break


if __name__ == "__main__":
    if False:
        # seems slower than CPU; to be reviewed
        torchDeviceName = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        torchDeviceName = "cpu"
    torchDevice = torch.device(torchDeviceName)
    print(f"Torch: {torch.__version__}, device: {torchDevice}")

    rules = Rules(4)
    board = Board(6, 7)

    if True:
        modelClass = SimpleModel
        modelParams = {
            'numInputs': board.numCells(),
            'numOutputs': board.width,
            'hiddenLayersNumFeatures': 50,
            'numHiddenLayers': 1,
        }

    else:
        modelClass = ConvModel
        modelParams = {
            'numInputs': board.numCells(),
            'numOutputs': board.width,
        }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "debugLog":
        model = RandomModel(torchDevice, board.width)
        simulation = Simulation(rules, board, model)
        simulation.run()
        simulation.debugLog()

    elif cmd == "debugReturns":
        model = initModel(modelClass, torchDevice=torchDevice)
        trainer = Trainer(rules, board, model)
        trainer.debugReturns()

    elif cmd == "train":
        model = initModel(modelClass, modelParams=modelParams, torchDevice=torchDevice)
        trainer = Trainer(rules, board, model)
        trainer.train()
        saveModel(model, modelParams)
        saveRewards(trainer.expectedReturnHistory)

    elif cmd == "play":
        model = initModel(modelClass, torchDevice=torchDevice)
        simulation = Simulation(rules, board, model)
        simulation.play()

    else:
        print("Unknown cmd")