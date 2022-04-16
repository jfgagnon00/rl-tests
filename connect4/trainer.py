import torch

from rules import *
from simulation import *
from reward import *


class Trainer:
    def __init__(self, rules, board, model, learningRate = 1e-6, gamma = 0.8):
        self._gamma = gamma
        self._simulation = Simulation(rules, board, model)
        self._optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    def train(self, episodes=30000):
        print("Start new training")

        self._simulation.model.train()

        for e in range(episodes):
            self._simulation.reset()
            self._simulation.run()

            if e % 100 == 0:
                print(f"Winner: {Rules.colorName(self._simulation.winColor)}")

            expectedReturnGrads, lenT = self._trainTrajectory(Rules.ColorRed)
            if e % 100 == 0:
                print(f"{e:>5d} - r - Grad: {expectedReturnGrads:>8.2f}, Len: {lenT:>3d}")

            expectedReturnGrads, lenT = self._trainTrajectory(Rules.ColorBlack)
            if e % 100 == 0:
                print(f"{e:>5d} - b - Grad: {expectedReturnGrads:>8.2f}, Len: {lenT:>3d}")

            if e % 100 == 0:
                print()

    def debugReturns(self):
        self._simulation.reset()
        self._simulation.run()

        print(f"startColor: {Rules.colorName(self._simulation.startColor)}, winColor: {Rules.colorName(self._simulation.winColor)}")
        print(f"Gamma: {self._gamma}")
        print(f"NumSteps: {self._simulation.numSteps}")

        self._debugReturns(Rules.ColorBlack)
        self._debugReturns(Rules.ColorRed)
        print()

    def _debugReturns(self, color):
        T = self._simulation.trajectories[color]

        expectedReturns = []
        for t in range(len(T)):
            r = Reward.Return(T, t, self._gamma)
            expectedReturns.append(r)

        if False:
            # normalize expectedReturns
            expectedReturns = torch.tensor(expectedReturns, requires_grad=True)
            expectedReturns = (expectedReturns - expectedReturns.mean()) / (expectedReturns.std() + 1e-9)

        expectedReturnGrads = 0
        for t in range(len(T)):
            grad = -T[t].logProbAction * expectedReturns[t]
            expectedReturnGrads += grad

        print(f"{Rules.colorName(color)}")
        print(f"    Total Grads: {expectedReturnGrads}")
        for t in range(len(T)):
            ts = T[t]
            print(f"    t: {t + 1:>3d}, G: {expectedReturns[t]:>8.2f}, r: {ts.reward:>5.1f}, logProb: {-ts.logProbAction:>6.3f}, c: {ts.column}, {Rules.applyName(ts.applyResult)}")

    def _trainTrajectory(self, color):
        T = self._simulation.trajectories[color]

        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        expectedReturns = []
        for t in range(len(T)):
            r = Reward.Return(T, t, self._gamma)
            expectedReturns.append(r)

        expectedReturns = torch.tensor(expectedReturns, requires_grad=True)

        if False:
            # normalize expectedReturns
            expectedReturns = (expectedReturns - expectedReturns.mean()) / (expectedReturns.std() + 1e-9)

        expectedReturnGrads = []
        for t in range(len(T)):
            grad = -T[t].logProbAction * expectedReturns[t]
            expectedReturnGrads.append(grad)

        self._optimizer.zero_grad()
        loss = torch.stack(expectedReturnGrads).sum()
        expectedReturnGrads = loss.item()
        loss.backward()
        self._optimizer.step()

        return expectedReturnGrads, len(T)
