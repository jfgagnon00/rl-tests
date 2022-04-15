from importlib.metadata import distribution
import random
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
            int(Rules.ColorBlack): [],
            int(Rules.ColorRed): [],
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
            # an action is actually in which column to play token
            column = action.item()

            # get reward and result of action
            applyResult = self.rules.apply(self.board, column, currentColor)
            reward = Reward.Get(applyResult)

            # log everything in the simulation
            logProbAction = distribution.log_prob(action).item()
            trajectoryStep = TrajectoryStep(column, logProbAction, reward, applyResult)
            self.trajectories[int(currentColor)].append(trajectoryStep)

            if applyResult == Rules.ApplyInvalid:
                # if current player selected an invalid move
                # start over with same player
                continue

            if applyResult == Rules.ApplyTie:
                self.winColor = Rules.ColorNone
                break

            if applyResult > Rules.ApplyTie:
                # color has won, stop
                self.winColor = currentColor

                # mark last action of opponent as loser
                self.trajectories[int(-currentColor)][-1].reward = -reward
                break

            currentColor = -currentColor

    def print(self):
        self.printHeader()
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

            # column, logProbAction, reward, applyResult
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
                # if current player selected an invalid move
                # start over with same player
                continue

            if trajectoryStep.applyResult == Rules.ApplyTie:
                # color has won, stop
                if Rules.ColorNone != self.winColor:
                    print("Error result tie !!!!!")
                break

            if trajectoryStep.applyResult > Rules.ApplyTie:
                # color has won, stop
                if color != self.winColor:
                    print("Error result winColor !!!!!")
                break

            color = -color

        if numSteps != self.numSteps:
            print("Error result numSteps !!!!!")

        self.printHeader()

    def printHeader(self):
        print(f"startColor: {Rules.colorName(self.startColor)}")
        print(f"winColor: {Rules.colorName(self.winColor)}")
        print(f"steps: {self.numSteps}")
