from importlib.metadata import distribution
import random
import sys
import torch

from board import *
from rules import *
from model import *
from reward import *

def DebugSimulate(rules, board):
    print("Starting")
    print(str(board))
    print("")

    column = random.randrange(0, board.width)
    currentColor = Rules.ColorRed

    while True:
        applyResult = rules.apply(board, column, currentColor)
        if applyResult == Rules.ApplyInvalid:
            continue

        print(f"{Rules.colorName(currentColor)} {column} -> {Rules.applyName(applyResult)}")
        print(str(board))
        print("")

        if applyResult > Rules.ApplyInconclusive:
            break

        column = random.randrange(0, board.width)
        currentColor = -currentColor

def GetTrajectories(rules, board, model):
    trajectories = {
        int(Rules.ColorBlack): [],
        int(Rules.ColorRed): []
    }

    currentColor = Rules.ColorRed if random.randrange(0, 1) > 0.5 else Rules.ColorBlack

    while True:
        state = board.cells.flatten()
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = model(state)

        distribution =  torch.distributions.Categorical(probabilities)
        action = distribution.sample()
        column = action.item()
        logProbAction = distribution.log_prob(action).item()

        applyResult = rules.apply(board, column, currentColor)
        reward = Reward(applyResult)
        trajectories[int(currentColor)].append( (column, logProbAction, reward, applyResult) )

        if applyResult == Rules.ApplyInvalid:
            continue

        if applyResult > Rules.ApplyInconclusive:
            break

        currentColor = -currentColor

    return trajectories

def Train(rules,
    board,
    learningRate = 1e-6,
    gamma = 0.9,
    episodes = 1):

    model = Model(board.numCells(), board.width, board.width + 2, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for e in range(episodes):
        board.reset()
        trajectories = GetTrajectories(rules, board, model)

        # grad = 0 # suppose etre un vecteur
        # for T in trajectories.values:
        #     for t in range(len(T)):
        #         r = Return(T, t, gamma)

        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

        # loss = criterion(pred, y)
        # if e % 100 == 0:
        #     print(e, loss.item())

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


if __name__ == "__main__":
    rules = Rules(4)
    board = Board(6, 7)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "sim":
        DebugSimulate(rules, board)

    if cmd == "train":
        Train(rules, board)
