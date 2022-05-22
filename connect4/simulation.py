import math
import random
import torch

from board import *
from scoped_perfcounter import *
from rules import *
from reward import *
from trajectory_step import *


class Replay:
    def __init__(self):
        self.startColor = Rules.ColorNone
        self.winColor = Rules.ColorNone
        self.lastColor = Rules.ColorNone
        self.lastApplyResult = Rules.ApplyInconclusive
        self.numSteps = 0
        self.trajectories = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }
        self.evalDt = 0.0
        self.choiceDt = 0.0
        self.applyRulesDt = 0.0
        self.trajectoryDt = 0.0


class Simulation:
    def randomStartColor():
        return Rules.ColorRed if random.randrange(0, 1) > 0.5 else Rules.ColorBlack

    def __init__(self, winningStreak, boardWidth, boardHeight):
        self._rules = Rules(winningStreak)
        self._board = Board(boardWidth, boardHeight)
        self._evalCounter = ScopedPerfCounter()
        self._choiceCounter = ScopedPerfCounter()
        self._applyRulesCounter = ScopedPerfCounter()
        self._trajectoryCounter = ScopedPerfCounter()

    def reset(self):
        self._board.reset()
        self._evalCounter.reset()
        self._choiceCounter.reset()
        self._applyRulesCounter.reset()
        self._trajectoryCounter.reset()

    def run(self, blackModel=None, redModel=None, startColor=Rules.ColorNone):
        replay = Replay()

        playHuman = lambda: self._humanPlayer(replay)
        playBackModel = lambda: self._modelPlayer(blackModel, Rules.ColorBlack)
        playRedModel = lambda: self._modelPlayer(redModel, Rules.ColorRed)
        players = {
            Rules.ColorBlack: playHuman if blackModel is None else playBackModel,
            Rules.ColorRed: playHuman if redModel is None else playRedModel,
        }

        if startColor == Rules.ColorNone:
            startColor = Simulation.randomStartColor()

        replay.lastColor = replay.startColor = startColor

        while True:
            replay.numSteps += 1

            # get player move
            column, logProbAction = players[replay.lastColor]()

            # get result of action
            with self._applyRulesCounter:
                replay.lastApplyResult = self._rules.apply(self._board, column.item(), replay.lastColor)

            # log everything
            with self._trajectoryCounter:
                reward = Reward.Get(replay.lastApplyResult)
                trajectoryStep = TrajectoryStep(column, logProbAction, reward, replay.lastApplyResult)
                replay.trajectories[replay.lastColor].append(trajectoryStep)

            if replay.lastApplyResult == Rules.ApplyInvalid:
                # invalid move, stop
                break

            if replay.lastApplyResult == Rules.ApplyTie:
                replay.winColor = Rules.ColorNone

                # give part of reward to opponent as well
                replay.trajectories[-replay.lastColor][-1].reward = reward
                break

            if replay.lastApplyResult > Rules.ApplyTie:
                # color has won, stop
                replay.winColor = replay.lastColor

                # since oppenent lost, penalize its last move
                replay.trajectories[-replay.lastColor][-1].reward = -reward
                break

            replay.lastColor = -replay.lastColor

        replay.evalDt += self._evalCounter.total()
        replay.choiceDt += self._choiceCounter.total()
        replay.applyRulesDt = self._applyRulesCounter.total()
        replay.trajectoryDt = self._trajectoryCounter.total()

        return replay

    def debugLog(self):
        self._printHeader()
        print()

        self._board.reset()
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
            testApply = self._rules.apply(self._board, trajectoryStep.column, color)
            if testApply == Rules.ApplyInvalid:
                # invalid move, forget about it and replay
                numSteps -= 1
                continue

            print(self._board)
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

    def _humanPlayer(self, replay):
        # display _board for player
        print()
        print(self._board)

        opponent = -replay.lastColor
        T = replay.trajectories[opponent]
        print(f"{Rules.colorName(opponent)}: {T[-1].column}")

        column = self._getInput()
        print()

        action = torch.tensor([column], dtype=torch.int32)
        # action.share_memory_()

        logProbAction = torch.tensor([0.0], dtype=torch.float32)
        # logProbAction.share_memory_()

        return action, logProbAction

    def _modelPlayer(self, model, player):
        with self._evalCounter:
            if Parameters.OpenAIState:
                empty_positions = np.where(self._board.cells == Rules.ColorNone, 1, 0)
                player_chips   = np.where(self._board.cells == player, 1, 0)
                opponent_chips = np.where(self._board.cells == -player, 1, 0)
                cells = np.array([empty_positions, player_chips, opponent_chips])
            else:
                cells = self._board.cells

            cells = cells.reshape((1, model.numInputs)).astype(np.float32)
            cells = torch.from_numpy(cells).to(model.device)

            actionProbabilities = model(cells)
            actionProbabilities

        # model gives probabilities per action reinforcement
        # learning needs to randomly choose from a
        # dristribution matching those actionProbabilities
        # (dunno why yet); that is, it is not classification
        # problem
        with self._choiceCounter:
            distribution = torch.distributions.Categorical(actionProbabilities)
            action = distribution.sample()
            # torch.distributions.Categorical.sample|log_prob are slow
            # so replace by actionProbabilities
            logProbAction = distribution.log_prob(action)
            # logProbAction = math.log(actionProbabilities[column])

        # action.share_memory_()
        # logProbAction.share_memory_()

        return action, logProbAction

    def _printHeader(self):
        print(f"startColor: {Rules.colorName(self.startColor)}")
        print(f"winColor: {Rules.colorName(self.winColor)}")
        print(f"steps: {self.numSteps}")

    def _getInput(self):
        numOutputs = self._board.width - 1
        while True:
            try:
                data = input(f"Your turn [0, {numOutputs}]> ")
                column = int(data)
                if 0 <= column and column <= numOutputs:
                    return column
            finally:
                pass