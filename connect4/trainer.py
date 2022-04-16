import torch

from rules import *
from simulation import *
from reward import *


class Trainer:
    def __init__(self, rules, board, model, learningRate = 1e-6, gamma = 0.8):
        self._gamma = gamma
        self._simulation = Simulation(rules, board, model)
        self._optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    def train(self, episodes=1000):
        print("Start new training")

        self._simulation.model.train()

        for e in range(episodes):
            self._simulation.reset()
            self._simulation.run()

            if e % 100 == 0:
                print(f"Winner: {Rules.colorName(self._simulation.winColor)}")

            expectedReturn, lenT = self._trainTrajectory(Rules.ColorRed)
            if e % 100 == 0:
                print(f"{e:>5d} - r - E[r(t)]: {expectedReturn:>8.2f}, Len: {lenT:>3d}")

            expectedReturn, lenT = self._trainTrajectory(Rules.ColorBlack)
            if e % 100 == 0:
                print(f"{e:>5d} - b - E[r(t)]: {expectedReturn:>8.2f}, Len: {lenT:>3d}")

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

        # a tad different than _trainTrajectory
        expectedReturn = []
        expectedReturnGrad = []
        expectedReturnGrads = 0
        for t in range(len(T)):
            r = Reward.Return(T, t, self._gamma)
            expectedReturn.append(r)

            grad = -T[t].logProbAction * r
            expectedReturnGrads += grad
            expectedReturnGrad.append(grad)

        print(f"{Rules.colorName(color)}")
        print(f"    Total Grads: {expectedReturnGrads}")
        for t in range(len(T)):
            ts = T[t]
            print(f"    t: {t:>3d}, G: {expectedReturn[t]:>8.2f}, r: {ts.reward:>5.1f}, c: {ts.column}, {Rules.applyName(ts.applyResult)}")

    def _trainTrajectory(self, color):
        T = self._simulation.trajectories[color]

        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        returns = []
        expectedReturn = 0
        for t in range(len(T)):
            r = Reward.Return(T, t, self._gamma)
            returns.append(r)
            expectedReturn += T[t].logProbAction * r

        # normalize returns
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        grads = []
        for t in range(len(T)):
            grad = -T[t].logProbAction * returns[t]
            grads.append(grad)

        self._optimizer.zero_grad()
        loss = torch.stack(grads).sum()
        loss.backward()
        self._optimizer.step()

        return expectedReturn, len(T)
