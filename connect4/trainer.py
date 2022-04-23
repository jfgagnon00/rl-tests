import torch

from parameters import *
from reward import *
from rules import *
from scoped_perfcounter import *
from simulation import *
from torch import multiprocessing as mp


class Episode:
    def __init__(self):
        self.replays = []

    def wins(self, replay):
        wins = {
            Rules.ColorBlack: 0,
            Rules.ColorRed: 0,
            Rules.ColorNone: 0, # means no winner, a tie
        }

        for r in self.replays:
            wins[r.winColor] += 1

        return wins

    def evalDtMs(self):
        iter = map(lambda r: r.evalDt, self.replays)
        return Episode._sumMs(iter)

    def choiceDtMs(self):
        iter = map(lambda r: r.choiceDt, self.replays)
        return Episode._sumMs(iter)

    def applyRulesDtMs(self):
        iter = map(lambda r: r.applyRulesDt, self.replays)
        return Episode._sumMs(iter)

    def trajectoryDtMs(self):
        iter = map(lambda r: r.trajectoryDt, self.replays)
        return Episode._sumMs(iter)

    def _sumMs(iter):
        return sum(iter) * 1000.0


class Trainer:
    _TrainingColor = Rules.ColorBlack

    def __init__(self, model, parametersClass):
        self._parameters = parametersClass
        self._model = model
        self._optimizer = torch.optim.RMSprop(model.parameters(), lr=parametersClass.LearningRate)
        self._gatherCounter = ScopedPerfCounter()
        self._backPropCounter = ScopedPerfCounter()
        self._processCount = mp.cpu_count() - 1
        self._processRange = range(self._processCount)
        self._processPool = None

    def startProcessPool(self):
        if self._parameters.UseMultiprocessing:
            self._processPool = mp.Pool(self._processCount,
                Trainer._initProcess,
                (self._parameters.WinningStreak, self._parameters.BoardWidth, self._parameters.BoardHeight))
        return self

    def stopProcessPool(self):
        if self._processPool is not None:
            self._processPool.close()
            self._processPool.join()
            del self._processPool
            self._processPool = None

    def train(self, saveFn=None):
        global localSimulation
        localSimulation = Simulation(self._parameters.WinningStreak,
            self._parameters.BoardWidth,
            self._parameters.BoardHeight)

        # one entry per episode
        self._gatherCounter.reset()
        self._backPropCounter.reset()

        expectedReturnsHistory = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }
        eLen = 0
        gameCount = 0
        colors = [Rules.ColorBlack, Rules.ColorRed]

        if self._processPool is not None:
            print("Waiting for process pool to init")
            self._waitProcessPoolInit()

        print("Start training")
        for eIndex in range(self._parameters.Episodes):
            with self._gatherCounter:
                e = self._gatherEpisode()

            eLen += 1
            gameCount += len(e.replays)

            with self._backPropCounter:
                for c in colors:
                    expectedReturn = self._trainTrajectories(e.replays, c)
                    expectedReturnsHistory[c].append(expectedReturn)

            if (eIndex == self._parameters.Episodes - 1) or (eIndex % self._parameters.LogEpisodeEveryN == 0):
                print(f"iteration {eIndex} - {gameCount} games simulated")

                for c in colors:
                    colorName = Rules.colorName(c)
                    meanExpectedReturn = np.mean(expectedReturnsHistory[c][-20:])
                    print(f"    {colorName:>14}: {meanExpectedReturn:>6.2f} Mean Exp. Ret")

                print(f"    Gather Episode: {self._gatherCounter.totalMs()/eLen:>5.3f} ms/game")
                print(f"          Backprop: {self._backPropCounter.totalMs()/eLen:>5.3f} ms/game")
                print()

                eLen = 0
                self._gatherCounter.reset()
                self._backPropCounter.reset()

            if (eIndex == self._parameters.Episodes - 1) or (eIndex % self._parameters.SaveEveryN == 0):
                if saveFn is not None:
                    saveFn(expectedReturnsHistory)

                expectedReturnsHistory = {
                    Rules.ColorBlack: [],
                    Rules.ColorRed: [],
                }

    def debugReturns(self):
        episode = self.__gatherCounterEpisodeInfo(1)
        simulation = episode["simulations"][-1]

        print(f"startColor: {Rules.colorName(simulation.startColor)}, winColor: {Rules.colorName(simulation.winColor)}, lastColor: {Rules.colorName(simulation.lastColor)}")
        print(f"Gamma: {self._gamma}")
        print(f"NumSteps: {simulation.numSteps}")

        self._debugReturns(simulation, Rules.ColorBlack)
        self._debugReturns(simulation, Rules.ColorRed)
        print()

    def _debugReturns(self, simulation, color):
        T = simulation.trajectories[color]
        _, expectedReturns = self._getTrajectoriesInfos([T], color)
        expectedReturn = np.array(expectedReturns).mean()

        print(f"{Rules.colorName(color)}")
        print(f"    Expected Return: {expectedReturn}")
        for t in range(len(T)):
            ts = T[t]
            print(f"    t: {t + 1:>2d}, Return: {expectedReturns[t]:>8.2f}, Reward: {ts.reward:>5.2f}, logProb: {-ts.logProbAction:>6.3f}, c: {ts.column}, {Rules.applyName(ts.applyResult)}")

    def _trainTrajectories(self, replays, color):
        gradients, expectedReturns = self._getTrajectoriesInfos(replays, color)

        if self._optimizer is not None:
            self._model.train()
            self._optimizer.zero_grad()
            gradients = torch.tensor(gradients, requires_grad=True)
            gradients.mean().backward()
            self._optimizer.step()

        return np.array(expectedReturns).mean()

    def _getTrajectoriesInfos(self, replays, color):
        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        gradients = []
        expectedReturns = []

        for r in replays:
            T = r.trajectories[color]
            rangeT = range(len(T))

            returnsT = [Reward.Return(T, t, self._parameters.Gamma) for t in rangeT]
            returnT = np.array(returnsT).sum()
            expectedReturns.append(returnT)

            gradientsT = [-T[t].logProbAction * returnsT[t] for t in rangeT]
            gradientsT = np.array(gradientsT)

            # normalize gradients
            if self._parameters.NormalizeGradients:
                gradientsT = (gradientsT - gradientsT.mean()) / (gradientsT.std() + 1e-9)

            gradientT = gradientsT.sum()
            gradients.append(gradientT)

        return gradients, expectedReturns

    def _gatherEpisode(self):
        episode = Episode()

        if self._processPool is None:
            self._model.eval()
            replay = Trainer._localSimulate(self._model)
            episode.replays.append(replay)

            for _ in self._processRange:
                replay = Trainer._localSimulate(self._model)
                episode.replays.append(replay)
        else:
            # start simulations on external processes
            remoteReplays = self._processPool.map_async(Trainer._localSimulate, [self._model for i in self._processRange])

            # keep some work for ourselves
            replay = Trainer._localSimulate(self._model)
            episode.replays.append(replay)

            # sync external processes
            remoteReplays.wait()
            for r in remoteReplays.get():
                episode.replays.append(r)

        return episode

    def __enter__(self):
        return self.startProcessPool()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stopProcessPool()

    def _initProcess(winningStreak, boardWidth, boardHeight):
        global localSimulation
        localSimulation = Simulation(winningStreak, boardWidth, boardHeight)

    def _waitProcessPoolInit(self):
        barrier = mp.Manager().Barrier(self._processCount)
        self._processPool.map(Trainer._waitProcessInit, [barrier for _ in self._processRange])
        del barrier

    def _waitProcessInit(barrier):
        barrier.wait()

    def _localSimulate(model):
        global localSimulation
        localSimulation.reset()
        return localSimulation.run(blackModel=model, redModel=model)
