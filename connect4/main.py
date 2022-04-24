import json
import os.path
import sys

from board import *
from conv_model import *
from parameters import *
from random_model import *
from rules import *
from simple_model import *
from simulation import *
from torch import multiprocessing as mp
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

    device = None
    if "device" in modelParams:
        device =  modelParams["device"]
        modelParams.pop("device", None)

    with open(f"{filename}.json", 'w') as f:
        json.dump(modelParams, f)

    if device is not None:
        modelParams["device"] = device

    model.save(filename)

def saveRewards(expectedReturnHistory):
    expectedReturnHistory = {
        Rules.colorName(Rules.ColorBlack): expectedReturnHistory[Rules.ColorBlack],
        Rules.colorName(Rules.ColorRed): expectedReturnHistory[Rules.ColorRed],
    }

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
        mp.set_start_method('spawn')
    else:
        torchDeviceName = "cpu"
    torchDevice = torch.device(torchDeviceName)
    print(f"Torch: {torch.__version__}, device: {torchDevice}")

    if True:
        modelClass = SimpleModel
        modelParams = {
            'numInputs': Parameters.BoardWidth * Parameters.BoardHeight,
            'numOutputs': Parameters.BoardWidth,
            'hiddenLayersNumFeatures': 50,
            'numHiddenLayers': 1,
        }

    else:
        modelClass = ConvModel
        modelParams = {
            'numInputs': Parameters.BoardWidth * Parameters.BoardHeight,
            'numOutputs': Parameters.BoardWidth,
        }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "debugLog":
        model = RandomModel(torchDevice, Parameters.BoardWidth)
        simulation = Simulation(Parameters.WinningStreak, Parameters.BoardWidth, Parameters.BoardHeight)
        simulation.run(model)
        simulation.debugLog()

    elif cmd == "debugReturns":
        model = initModel(modelClass, torchDevice=torchDevice)
        with Trainer(model, Parameters) as trainer:
            trainer.debugReturns()

    elif cmd == "train":
        model = initModel(modelClass, modelParams=modelParams, torchDevice=torchDevice)
        trainer = Trainer(model, Parameters)
        with trainer:
            def save(expectedReturnsHistory):
                saveModel(model, modelParams)
                saveRewards(expectedReturnsHistory)
            trainer.train(saveFn=save)
        del trainer
        del model

    elif cmd == "play":
        model = initModel(modelClass, torchDevice=torchDevice)
        simulation = Simulation(Parameters.WinningStreak, Parameters.BoardWidth, Parameters.BoardHeight)
        replay = simulation.run(blackModel=model)

        # print winning move
        print(simulation._board)
        T = replay.trajectories[replay.lastColor]
        print(f"{Rules.colorName(replay.winColor)}: {T[-1].column}, {Rules.applyName(replay.lastApplyResult)}")

    else:
        print("Unknown cmd")