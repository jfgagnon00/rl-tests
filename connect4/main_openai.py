import json
import os.path
import random

import gym
import gym_connect4
import numpy as np
import torch
import torch.distributions
import torch.optim

from rules import *
from parameters import *
from random_model import *
from simple_model import *


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

def actionFromProbs(actionProbabilities):
    # model gives probabilities per action reinforcement
    # learning needs to randomly choose from a
    # dristribution matching those actionProbabilities
    # (dunno why yet); that is, it is not classification
    # problem
    distribution = torch.distributions.Categorical(actionProbabilities)
    action = distribution.sample()
    column = int(action.item())
    # torch.distributions.Categorical.sample|log_prob are slow
    # so replace by actionProbabilities
    logProbAction = distribution.log_prob(action)
    # logProbAction = math.log(actionProbabilities[column])

    return column, logProbAction

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

def saveHistory(history):
    i = 0
    while True:
        filename = f"history{i}.json"

        if os.path.exists(filename):
            i += 1
            continue

        with open(filename, "w") as f:
            json.dump(history, f)

        break

def train(model, optimizer, batch, gamma=0.5):
    T = len(batch["rewards"])
    returns = np.empty(T, dtype=np.float32)
    future_ret = 0

    for t in reversed(range(T)):
        future_ret = batch["rewards"][t] + gamma * future_ret
        returns[t] = future_ret

    returns = torch.tensor(returns, requires_grad=True)
    returns = (returns - returns.mean()) # / (returns.std() + 1e-9)

    logProbs = torch.tensor(batch["logProbs"], requires_grad=True)
    loss = torch.dot(-logProbs, returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


if __name__ == "__main__":
    if False:
        # seems slower than CPU; to be reviewed
        torchDeviceName = "cuda:0" if torch.cuda.is_available() else "cpu"
        mp.set_start_method('spawn')
    else:
        torchDeviceName = "cpu"
    torchDevice = torch.device(torchDeviceName)
    print(f"Torch: {torch.__version__}, device: {torchDevice}")

    modelClass = SimpleModel
    modelParams = {
        'numInputs': Parameters.BoardWidth * Parameters.BoardHeight * 3, # * 3 due to how states are encoded
        'numOutputs': Parameters.BoardWidth,
        'hiddenLayersNumFeatures': 64,
        'numHiddenLayers': 2,
    }
    model = initModel(modelClass, modelParams=modelParams, torchDevice=torchDevice)
    model.train()

    randomModel = RandomModel(torchDevice, modelParams["numInputs"], modelParams["numOutputs"])
    opponent = model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with gym.make("Connect4-v0") as env:
        history = {
            "loss": [],
            "totalRewards": [],
        }

        for e in range(100000):
            states = env.reset()

            info = {
                "legal_actions": env.get_moves()
            }

            batch = {
                "rewards": [],
                "probsModel": [],
                "actionsModel": [],
                "probsModelOppnenent": [],
                "actionsOpponent": [],
                "logProbs": [],
            }

            while True:
                if env.winner is not None:
                    break

                x = states[0].flatten().reshape((1, model.numInputs)).astype(np.float32)
                x = torch.from_numpy(x)
                actionProbs = model(x)

                action, actionLogProb = actionFromProbs(actionProbs)
                if action in info["legal_actions"]:
                    states, rewards, done, info = env.step(action)
                else:
                    rewards = (-10, -10)
                    done = True

                batch["logProbs"].append(actionLogProb)
                batch["probsModel"].append(actionProbs)
                batch["actionsModel"].append(action)
                batch["rewards"].append(rewards[0])

                if done:
                    break

                x = states[0].flatten().reshape((1, opponent.numInputs)).astype(np.float32)
                x = torch.from_numpy(x)
                actionProbs = opponent(x)

                action, _ = actionFromProbs(actionProbs)
                if action not in info["legal_actions"]:
                    while True:
                        action = random.randint(0, Parameters.BoardWidth - 1)
                        if action in info["legal_actions"]:
                            break

                batch["probsModelOppnenent"].append(actionProbs)
                batch["actionsOpponent"].append(action)

                states, rewards, done, info = env.step(action)

            loss = train(model, optimizer, batch)
            totalReward = sum(batch["rewards"])

            history["loss"].append(loss.item())
            history["totalRewards"].append(totalReward)

            l = history["loss"]
            r = history["totalRewards"]

            if  len(l) > 0 and len(l) % 100 == 0:
                L = np.mean(l[-100])
                R = np.mean(r[-100])
                print(f"e: {e:>4d} - mean total reward = {R:>8.4f}, mean loss = {L:>8.4f}")

            if e % 1000 == 0:
                saveModel(model, modelParams)
                saveHistory(history)
                history = {
                    "loss": [],
                    "totalRewards": [],
                }

    saveModel(model, modelParams)
    saveHistory(history)