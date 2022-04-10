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
        # make current color choose its action based on
        # the state of the board
        state = board.cells.flatten()
        state = torch.from_numpy(state).float().unsqueeze(0)

        # all colors have the same model, so no distinction
        # per color is needed
        actionProbabilities = model(state)

        # model gives probabilities per action reinforcement
        # learning needs to randomly choose from a
        # dristribution matching those actionProbabilities
        # (dunno why yet); that is, it is not classification
        # problem
        distribution =  torch.distributions.Categorical(actionProbabilities)
        action = distribution.sample()
        # an action is actually in which column to play token
        column = action.item()

        # get reward and result of action
        applyResult = rules.apply(board, column, currentColor)
        reward = Reward(applyResult)

        # log everything in the trajectories
        logProbAction = distribution.log_prob(action).item()
        trajectories[int(currentColor)].append( (column, logProbAction, reward, applyResult) )

        if applyResult == Rules.ApplyInvalid:
            # if current player selected an invalid move
            # start over with same player
            continue

        if applyResult > Rules.ApplyInconclusive:
            # color has won, stop
            break

        currentColor = -currentColor

    return trajectories

def Train(rules,
    board,
    learningRate = 1e-6,
    gamma = 0.9,
    episodes = 1000):

    model = Model(board.numCells(), board.width, board.width + 2, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for e in range(episodes):
        board.reset()
        trajectories = GetTrajectories(rules, board, model)

        # book equation is unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        returns = []
        for T in trajectories.values():
            for t in range(len(T)):
                r = Return(T, t, gamma)
                returns.append(r)

        # normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        grads = []
        for T in trajectories.values():
            for t in range(len(T)):
                # column, logProbAction, reward, applyResult
                _, logProbAction, _, _ = T[t]
                grad = -logProbAction * returns[t]
                grads.append(grad)

        optimizer.zero_grad()
        loss = torch.tensor(grads, requires_grad=True).sum()
        if e % 100 == 0:
            print(e, loss.item())
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "connect4_model.bin")


if __name__ == "__main__":
    rules = Rules(4)
    board = Board(6, 7)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "sim":
        DebugSimulate(rules, board)

    if cmd == "train":
        Train(rules, board)