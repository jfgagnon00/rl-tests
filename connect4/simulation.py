import random
import sys
import torch

from board import *
from rules import *
from model import *
from reward import *
from trajectory_step import *


class Simulation:
    def __init__(self, rules, board, model):
        self.rules = rules
        self.board = board
        self.model = model
        self.startColor = Rules.ColorNone
        self.winColor = Rules.ColorNone
        self.reset()

    def reset(self):
        self.board.reset()
        self.numSteps = 0
        self.trajectories = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }

    def run(self):
        currentColor = self.startColor = Rules.ColorRed if random.randrange(0, 1) > 0.5 else Rules.ColorBlack

        while True:
            self.numSteps += 1

            # make current color choose its action based on
            # the state of the board
            state = self.board.cells.flatten()
            state = torch.from_numpy(state).float().unsqueeze(0)

            # all colors have the same model, so no distinction
            # per color is needed
            actionProbabilities = self.model(state)

            # model gives probabilities per action reinforcement
            # learning needs to randomly choose from a
            # dristribution matching those actionProbabilities
            # (dunno why yet); that is, it is not classification
            # problem
            distribution = torch.distributions.Categorical(actionProbabilities)
            action = distribution.sample()
            logProbAction = distribution.log_prob(action).item()
            # an action is actually in which column to play token
            column = action.item()

            # get reward and result of action
            applyResult = self.rules.apply(self.board, column, currentColor)
            reward = Reward.Get(applyResult)

            # log everything in the simulation
            trajectoryStep = TrajectoryStep(column, logProbAction, reward, applyResult)
            self.trajectories[currentColor].append(trajectoryStep)

            if applyResult == Rules.ApplyInvalid or applyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone
                break

            if applyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = currentColor
                break

            currentColor = -currentColor

    def play(self):
        # comptuer always starts
        # comptuer is always black
        currentColor = self.startColor = Rules.ColorBlack

        while True:
            self.numSteps += 1

            if currentColor == Rules.ColorBlack:
                state = self.board.cells.flatten()
                state = torch.from_numpy(state).float().unsqueeze(0)

                actionProbabilities = self.model(state)
                column = torch.argmax(actionProbabilities)
            else:
                column = self._getInput()

            applyResult = self.rules.apply(self.board, column, currentColor)

            # log everything in the simulation
            trajectoryStep = TrajectoryStep(column, 0, 0, applyResult)
            self.trajectories[currentColor].append(trajectoryStep)

            print(f"{Rules.colorName(currentColor)} - {trajectoryStep.column}")
            print(Rules.applyName(trajectoryStep.applyResult))
            print(self.board)
            print()

            if applyResult == Rules.ApplyInvalid or applyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone
                break

            if applyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = currentColor
                break

            currentColor = -currentColor

    def debugLog(self):
        self._printHeader()
        print()

        self.board.reset()
        steps = {
            int(Rules.ColorBlack): 0,
            int(Rules.ColorRed): 0,
        }
        color = self.startColor
        numSteps = 0

        while True:
            numSteps += 1
            step = steps[color]

            trajectoryStep = self.trajectories[color][step]

            print(f"{Rules.colorName(color)} - {trajectoryStep.column}, reward: {trajectoryStep.reward}")
            print(Rules.applyName(trajectoryStep.applyResult))
            testApply = self.rules.apply(self.board, trajectoryStep.column, color)
            print(self.board)
            print()

            steps[color] += 1

            if trajectoryStep.applyResult != testApply:
                print("Error replay result !!!!!")

            if trajectoryStep.applyResult == Rules.ApplyInvalid or trajectoryStep.applyResult == Rules.ApplyTie:
                # color has won, stop
                if Rules.ColorNone != self.winColor:
                    print("Error result !!!!!")
                break

            if trajectoryStep.applyResult > Rules.ApplyTie:
                # color has won, stop
                if color != self.winColor:
                    print("Error result winColor !!!!!")
                break

            color = -color

        if numSteps != self.numSteps:
            print("Error result numSteps !!!!!")

        self._printHeader()

    def _printHeader(self):
        print(f"startColor: {Rules.colorName(self.startColor)}")
        print(f"winColor: {Rules.colorName(self.winColor)}")
        print(f"steps: {self.numSteps}")

    def _getInput(self):
        numOutputs = self.model.numOutputs - 1
        while True:
            try:
                data = input(f"Your turn [0, {numOutputs}]> ")
                column = int(data)
                if 0 <= column and column <= numOutputs:
                    return column
            finally:
                pass