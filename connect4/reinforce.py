import torch
import numpy as np

from algorithm import *
from rules import *

class Reinforce(Algorithm):

    def __init__(self, parameters):
        self._parameters = parameters
        self._epsilonThreshold = parameters.EpsilonThreshold

    def eval(self, cells, model, color):
        actionProbabilities = self._sample(cells, model, color)

        # reinforcement learning needs to randomly choose from
        # a dristribution matching those actionProbabilities
        # this is to balance exploration vs exploitation
        distribution = torch.distributions.Categorical(actionProbabilities)
        action = distribution.sample()
        logProbAction = distribution.log_prob(action)

        return action.item(), logProbAction

    def backward(self, simulationTrajectory):
        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

        lenT = len(simulationTrajectory.algorithmStates)
        rangeT = range(lenT)

        futureRet = 0
        returnsT = np.empty(lenT, dtype=np.float32)
        for t in reversed(rangeT):
            futureRet = simulationTrajectory.rewards[t] + self._parameters.Gamma * futureRet
            returnsT[t] = futureRet
        returns = torch.from_numpy(returnsT)

        logProbs = [simulationTrajectory.algorithmStates[t] for t in rangeT]
        logProbs = torch.cat(logProbs)

        gradient = torch.dot(-logProbs, returns - returns.mean()).view(1)
        gradient.mean().backward()

        return returnsT

    def update(self):
        self._epsilonThreshold *= self._parameters.EpsilonDecay

    def _sample(self, cells, model, color):
        if self._parameters.OpenAIState:
            empty_positions = np.where(cells == Rules.ColorNone, 1, 0)
            player_chips = np.where(cells == color, 1, 0)
            opponent_chips = np.where(cells == -color, 1, 0)
            cells = np.array([empty_positions, player_chips, opponent_chips])

        cells = cells.reshape((1, model.numInputs)).astype(np.float32)
        cells = torch.from_numpy(cells).to(model.device)

        return model(cells)
