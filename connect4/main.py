from ast import Mod
import random
import sys

from board import *
from rules import *
from model import *

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
        currentColor = 3 - currentColor

def Train(rules,
    board,
    learningRate = 1e-6,
    episodes = 10000):

    model = Model(board.numCells(), board.width, board.width + 2, 0)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    # adapt to reinforcement learning
    x = torch.linspace(-3, 3, board.numCells())
    y = torch.linspace(-3, 3, board.width)

    for e in range(episodes):
        pred = model(x)

        loss = criterion(pred, y)
        if e % 100 == 0:
            print(e, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    rules = Rules(4)
    board = Board(6, 7)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "sim":
        DebugSimulate(rules, board)

    if cmd == "train":
        Train(rules, board)
