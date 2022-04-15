import torch

from rules import *
from simulation import *
from reward import *


class Trainer:
    def __init__(self, rules, board, model, learningRate = 1e-6, gamma = 0.8):
        self._gamma = gamma
        self._simulation = Simulation(rules, board, model)
        self._optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    def train(self, episodes = 1000000):
        print("Start new training")

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

    def saveModel(self, filename="connect4_model.bin"):
        torch.save(self._simulation.model.state_dict(), filename)

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
