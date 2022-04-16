import torch

from rules import *
from simulation import *
from reward import *


class Trainer:
    k_NormalizeExpectedReturns =  False

    def __init__(self, rules, board, model, learningRate = 1e-6, gamma = 0.8):
        self._gamma = gamma
        self._simulation = Simulation(rules, board, model)
        self._optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
        self.totalRewardHistory = []

    def train(self, episodes=10000):
        print("Start new training")

        self.totalRewardHistory = []
        self._simulation.model.train()

        for e in range(episodes):
            self._simulation.reset()
            self._simulation.run()

            if self._simulation.lastApplyResult == Rules.ApplyInvalid:
                # game ended with an invalid move
                # pick color that played it
                color = self._simulation.lastColor
            elif self._simulation.lastApplyResult == Rules.ApplyTie:
                # game ended with a tie
                # pick random color
                color = Simulation.randomStartColor()
            else:
                # game ended with a clear win
                # pick winning color
                color = self._simulation.winColor

            totalRewards, totalExpectedReturnGrads, lenT = self._trainTrajectory(color)
            self.totalRewardHistory.append(totalRewards)

            if e % 100 == 0:
                colorName = Rules.colorName(color)
                applyResultName = Rules.applyName(self._simulation.lastApplyResult)
                print(f"e: {e:>5d} {colorName:5} - Len: {lenT:>3d}, Tot. Grad: {totalExpectedReturnGrads:>8.2f}, Tot. Reward: {totalRewards:>8.2f}, {applyResultName}")

    def debugReturns(self):
        self._simulation.reset()
        self._simulation.run()

        print(f"startColor: {Rules.colorName(self._simulation.startColor)}, winColor: {Rules.colorName(self._simulation.winColor)}, , lastColor: {Rules.colorName(self._simulation.lastColor)}")
        print(f"Gamma: {self._gamma}")
        print(f"NumSteps: {self._simulation.numSteps}")

        self._debugReturns(Rules.ColorBlack)
        self._debugReturns(Rules.ColorRed)
        print()

    def _debugReturns(self, color):
        expectedReturns, expectedReturnGrads, lenT = self._getExpectedInfos(color)
        totalExpectedReturnGrads = torch.stack(expectedReturnGrads).sum().item()

        print(f"{Rules.colorName(color)}")
        print(f"    Total Grads: {totalExpectedReturnGrads}")
        T = self._simulation.trajectories[color]
        for t in range(lenT):
            ts = T[t]
            print(f"    t: {t + 1:>2d}, Return: {expectedReturns[t]:>8.2f}, Reward: {ts.reward:>5.2f}, logProb: {-ts.logProbAction:>6.3f}, c: {ts.column}, {Rules.applyName(ts.applyResult)}")

    def _trainTrajectory(self, color):
        _, expectedReturnGrads, lenT = self._getExpectedInfos(color)

        T = self._simulation.trajectories[color]
        totalRewards = 0.0
        for t in range(lenT):
            totalRewards += T[t].reward

        self._optimizer.zero_grad()
        loss = torch.stack(expectedReturnGrads).sum()
        totalExpectedReturnGrads = loss.item()
        loss.backward()
        self._optimizer.step()

        return totalRewards, totalExpectedReturnGrads, lenT

    def _getExpectedInfos(self, color):
        T = self._simulation.trajectories[color]
        lenT = len(T)

        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        expectedReturns = []
        for t in range(lenT):
            r = Reward.Return(T, t, self._gamma)
            expectedReturns.append(r)

        expectedReturns = torch.tensor(expectedReturns, requires_grad=True)

        # normalize expectedReturns
        if Trainer.k_NormalizeExpectedReturns:
            expectedReturns = (expectedReturns - expectedReturns.mean()) / (expectedReturns.std() + 1e-9)

        expectedReturnGrads = []
        for t in range(lenT):
            grad = -T[t].logProbAction * expectedReturns[t]
            expectedReturnGrads.append(grad)

        return expectedReturns, expectedReturnGrads, lenT

