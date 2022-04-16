import random
import torch

from board import *
from rules import *
from reward import *
from trajectory_step import *


class Simulation:
    def randomStartColor():
        return Rules.ColorRed if random.randrange(0, 1) > 0.5 else Rules.ColorBlack

    def __init__(self, rules, board, model):
        self.rules = rules
        self.board = board
        self.model = model
        self.startColor = Rules.ColorNone
        self.reset()

    def reset(self):
        self.board.reset()
        self.winColor = Rules.ColorNone
        self.lastColor = Rules.ColorNone
        self.lastApplyResult = Rules.ApplyInconclusive
        self.numSteps = 0
        self.trajectories = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }

    def run(self, startColor=Rules.ColorNone):
        if startColor == Rules.ColorNone:
            startColor = Simulation.randomStartColor()

        self.lastColor = self.startColor = startColor

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
            self.lastApplyResult = self.rules.apply(self.board, column, self.lastColor)
            reward = Reward.Get(self.lastApplyResult)

            # log everything in the simulation
            trajectoryStep = TrajectoryStep(column, logProbAction, reward, self.lastApplyResult)
            self.trajectories[self.lastColor].append(trajectoryStep)

            if self.lastApplyResult == Rules.ApplyInvalid:
                # invalid move, replay
                continue

            if self.lastApplyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone
                break

            if self.lastApplyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = self.lastColor
                break

            self.lastColor = -self.lastColor

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
                column = torch.distributions.Categorical(actionProbabilities).sample().item()
            else:
                # display board for player
                print()
                print(self.board)
                column = self._getInput()
                print()

            applyResult = self.rules.apply(self.board, column, currentColor)

            # log everything in the simulation
            trajectoryStep = TrajectoryStep(column, 0, 0, applyResult)
            self.trajectories[currentColor].append(trajectoryStep)

            # log everything to console
            print(f"{Rules.colorName(currentColor):5}: {column} - {Rules.applyName(applyResult)}")

            if applyResult == Rules.ApplyInvalid:
                # invalid move, replay
                continue

            if applyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone
                print()
                print(self.board)
                break

            if applyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = currentColor
                print()
                print(self.board)
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

            if trajectoryStep.applyResult == Rules.ApplyInvalid:
                # invalid move, replay
                continue

            if trajectoryStep.applyResult == Rules.ApplyTie:
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