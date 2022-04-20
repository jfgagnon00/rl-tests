import math
import random
import torch

from board import *
from rules import *
from reward import *
from time import perf_counter
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
        self.evalModelDt = 0.0
        self.choiceDt = 0.0
        self.applyRulesDt = 0.0
        self.trajectoryDt = 0.0

    def run(self, startColor=Rules.ColorNone):
        if startColor == Rules.ColorNone:
            startColor = Simulation.randomStartColor()

        self.lastColor = self.startColor = startColor

        while True:
            self.numSteps += 1

            # all colors have the same model, so no distinction
            # per color is needed

            t0 = perf_counter()

            cells = torch.from_numpy(self.board.cells)
            actionProbabilities = self.model(cells)

            t01 = perf_counter()

            # model gives probabilities per action reinforcement
            # learning needs to randomly choose from a
            # dristribution matching those actionProbabilities
            # (dunno why yet); that is, it is not classification
            # problem
            distribution = torch.distributions.Categorical(actionProbabilities)
            action = distribution.sample()
            column = int(action.item())
            # torch.distributions.Categorical.sample|log_prob are slow
            # so replace by actionProbabilities
            # logProbAction = distribution.log_prob(action).item()
            logProbAction = math.log(actionProbabilities[column])

            t1 = perf_counter()

            # get reward and result of action
            self.lastApplyResult = self.rules.apply(self.board, column, self.lastColor)

            t2 = perf_counter()

            self.evalModelDt += t01 - t0
            self.choiceDt += t1 - t01
            self.applyRulesDt = t2 - t1

            if self.lastApplyResult == Rules.ApplyInvalid:
                # invalid move, forget about it and replay
                self.numSteps -= 1
                continue

            # log everything in the simulation
            reward = Reward.Get(self.lastApplyResult)
            trajectoryStep = TrajectoryStep(column, logProbAction, reward, self.lastApplyResult)
            self.trajectories[self.lastColor].append(trajectoryStep)

            t3 = perf_counter()
            self.trajectoryDt += t3 - t2

            if self.lastApplyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone

                # give part of reward to opponent as well
                self.trajectories[-self.lastColor][-1].reward = reward
                break

            if self.lastApplyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = self.lastColor

                # since oppenent lost, penalize its last move
                self.trajectories[-self.lastColor][-1].reward = -reward
                break

            self.lastColor = -self.lastColor

    def play(self):
        # comptuer always starts
        # comptuer is always black
        currentColor = self.startColor = Rules.ColorBlack

        while True:
            self.numSteps += 1

            if currentColor == Rules.ColorBlack:
                cells = torch.from_numpy(self.board.cells)
                actionProbabilities = self.model(cells)
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
            if testApply == Rules.ApplyInvalid:
                # invalid move, forget about it and replay
                numSteps -= 1
                continue

            print(self.board)
            print()

            steps[color] += 1

            if trajectoryStep.applyResult != testApply:
                print("Error replay result !!!!!")

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