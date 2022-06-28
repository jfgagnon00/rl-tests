import torch

from parameters import *
from reward import *
from rules import *
from scoped_perfcounter import *
from simulation import *
from algorithm import *


class Trainer:
    def __init__(self, parameters, algorithm, blackModel, redModel):
        self._parameters = parameters
        self._algorithm = algorithm
        self._blackModel = blackModel
        self._redModel = redModel

        self._simulation = Simulation(parameters.WinningStreak,
            parameters.BoardWidth,
            parameters.BoardHeight)

        self._optimizer = torch.optim.RMSprop(self._blackModel.parameters(),
            lr=parameters.LearningRate)

        self._simulateCounter = ScopedPerfCounter()
        self._trainCounter = ScopedPerfCounter()

    def train(self, saveFn=None):
        self._simulateCounter.reset()
        self._trainCounter.reset()

        expectedReturnsHistory = []
        replayNumSteps = []
        wins = 0.0
        draws = 0.0
        simulationCount = 0

        print("Start training")
        for eIndex in range(self._parameters.Episodes):
            with self._simulateCounter:
                self._blackModel.eval()
                self._simulation.reset()
                replay = self._simulation.run(self._blackModel, self._redModel, algorithm=self._algorithm)
                simulationCount += 1

            if self._optimizer is not None:
                with self._trainCounter:
                    self._blackModel.train()
                    self._optimizer.zero_grad()
                    returns = self._algorithm.backward(replay.trajectories[Rules.ColorBlack])
                    self._optimizer.step()
                    self._algorithm.update()

            expectedReturnsHistory.append(returns.mean().item())
            replayNumSteps.append(replay.numSteps)
            wins += 1 if replay.winColor == Rules.ColorBlack else 0
            draws += 1 if replay.winColor == Rules.ColorNone else 0

            if (eIndex == self._parameters.Episodes - 1) or (eIndex % self._parameters.LogEpisodeEveryN == 0):
                print(f"iteration {eIndex}")

                meanExpectedReturn = np.mean(expectedReturnsHistory[-simulationCount:])
                meanNumSteps = np.mean(replayNumSteps[-simulationCount:])
                meanWins = np.mean(wins / simulationCount)
                meanDraws = np.mean(draws / simulationCount)
                print(f"    {meanExpectedReturn:>6.2f} Mean Exp. Ret")
                print(f"    {meanNumSteps:>6.2f} Mean Sim Num Steps")
                print(f"    {meanWins:>6.2f} meanWins")
                print(f"    {meanDraws:>6.2f} meanDraws")

                print(f"    Sim time: {self._simulateCounter.totalMs()/simulationCount:>5.3f} ms/game")
                print(f"  Train time: {self._trainCounter.totalMs()/simulationCount:>5.3f} ms/game")
                print()

                simulationCount = 0
                wins = 0.0
                draws = 0.0
                self._simulateCounter.reset()
                self._trainCounter.reset()

            if (eIndex == self._parameters.Episodes - 1) or (eIndex % self._parameters.SaveEveryN == 0):
                if saveFn is not None:
                    saveFn(expectedReturnsHistory)