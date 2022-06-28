from operator import mod
import random
import torch

from algorithm import *
from board import *
from scoped_perfcounter import *
from rules import *
from reward import *

class Trajectory:
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.algorithmStates = []
        self.boardBeforeActionStates = []
        self.boardAfterActionStates = []

class Replay:
    def __init__(self):
        self.startColor = Rules.ColorNone
        self.winColor = Rules.ColorNone
        self.lastColor = Rules.ColorNone
        self.lastApplyResult = Rules.ApplyInconclusive
        self.numSteps = 0
        self.trajectories = {
            Rules.ColorBlack: Trajectory(),
            Rules.ColorRed: Trajectory(),
        }
        self.evalDt = 0.0
        self.applyRulesDt = 0.0


class Simulation:
    def randomStartColor():
        return Rules.ColorRed if random.randrange(0, 1) > 0.5 else Rules.ColorBlack

    def __init__(self, winningStreak, boardWidth, boardHeight):
        self._rules = Rules(winningStreak)
        self._board = Board(boardWidth, boardHeight)
        self._evalCounter = ScopedPerfCounter()
        self._applyRulesCounter = ScopedPerfCounter()

    def reset(self):
        self._board.reset()
        self._evalCounter.reset()
        self._applyRulesCounter.reset()

    def run(self, blackModel=None, redModel=None, startColor=Rules.ColorNone, algorithm=None):
        replay = Replay()

        playHuman = lambda: self._humanPlayer(replay)
        playBackModel = lambda: self._modelPlayer(blackModel, Rules.ColorBlack, algorithm)
        playRedModel = lambda: self._modelPlayer(redModel, Rules.ColorRed, algorithm)
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
            action, algorithmState = players[replay.lastColor]()

            boardBefore = self._board.clone()

            # get result of action
            with self._applyRulesCounter:
                replay.lastApplyResult = self._rules.apply(self._board, action, replay.lastColor)

            boardAfter = self._board.clone()

            # log everything
            reward = Reward.Get(replay.lastApplyResult)
            step = replay.trajectories[replay.lastColor]
            step.actions.append(action)
            step.rewards.append(reward)
            step.algorithmStates.append(algorithmState)
            step.boardBeforeActionStates.append(boardBefore)
            step.boardAfterActionStates.append(boardAfter)

            if replay.lastApplyResult == Rules.ApplyInvalid:
                # invalid move, stop
                break

            if replay.lastApplyResult == Rules.ApplyTie:
                replay.winColor = Rules.ColorNone

                # give part of reward to opponent as well
                replay.trajectories[-replay.lastColor].rewards[-1] = reward
                break

            if replay.lastApplyResult > Rules.ApplyTie:
                # color has won, stop
                replay.winColor = replay.lastColor

                # since oppenent lost, penalize its last move
                replay.trajectories[-replay.lastColor].rewards[-1] = -reward
                break

            replay.lastColor = -replay.lastColor

        replay.evalDt += self._evalCounter.total()
        replay.applyRulesDt = self._applyRulesCounter.total()

        return replay

    def _modelPlayer(self, model, color, algorithm):
        with self._evalCounter:
            return algorithm.eval(self._board.cells, model, color)

    def _humanPlayer(self, replay):
        # display _board for player
        print()
        print(self._board)

        opponent = -replay.lastColor
        T = replay.actions[opponent]
        print(f"{Rules.colorName(opponent)}: {T[-1]}")

        action = self._getInput()
        print()

        return action, None

    def _getInput(self):
        numOutputs = self._board.width - 1
        while True:
            try:
                data = input(f"Your turn [0, {numOutputs}]> ")
                action = int(data)
                if 0 <= action and action <= numOutputs:
                    return action
            finally:
                pass