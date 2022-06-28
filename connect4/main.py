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
from trainer import *
from reinforce import *
from dqn import *


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
    if not os.path.exists("history"):
        os.makedirs("history")

    i = 0
    while True:
        filename = f"history/expected-return-history{i}.json"

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
    torchDeviceName = "cpu"
    torchDevice = torch.device(torchDeviceName)
    print(f"Torch: {torch.__version__}, device: {torchDevice}")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        randomModel = RandomModel(torchDevice,
            Parameters.ModelParams["numInputs"],
            Parameters.ModelParams["numOutputs"])

        blackModel = initModel(Parameters.ModelClass,
            modelParams=Parameters.ModelParams,
            torchDevice=torchDevice)
        redModel = randomModel # initModel(modelClass, modelParams=modelParams, torchDevice=torchDevice)
        algorithm = Parameters.AlgorithmClass(**Parameters.AlgorithmParams)
        trainer = Trainer(Parameters, algorithm, blackModel, redModel)

        def save(expectedReturnsHistory):
            saveModel(blackModel, Parameters.ModelParams)
            saveRewards(expectedReturnsHistory)

        trainer.train(saveFn=save)

    elif cmd == "play":
        blackModel = initModel(Parameters.ModelClass, torchDevice=torchDevice)
        simulation = Simulation(Parameters.WinningStreak, Parameters.BoardWidth, Parameters.BoardHeight)
        replay = simulation.run(blackModel=blackModel)

        # print winning move
        print(simulation._board)
        T = replay.trajectories[replay.lastColor]
        print(f"{Rules.colorName(replay.winColor)}: {T[-1].column}, {Rules.applyName(replay.lastApplyResult)}")

    else:
        print("Unknown cmd")