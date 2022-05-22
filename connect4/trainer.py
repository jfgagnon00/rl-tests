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

    def __init__(self, trainModel, opponentModel, parametersClass):
        self._parameters = parametersClass
        self._model = trainModel
        self._opponentModel = opponentModel
        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=parametersClass.LearningRate)
        self._gatherCounter = ScopedPerfCounter()
        self._backPropCounter = ScopedPerfCounter()
        self._processCount = 0 # mp.cpu_count() - 1
        self._processRange = range(self._processCount)
        self._processPool = None

    def startProcessPool(self):
        if self._parameters.UseMultiprocessing and self._processCount > 0:
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
        wins = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }
        draws = {
            Rules.ColorBlack: [],
            Rules.ColorRed: [],
        }
        eLen = 0
        gameCount = 0
        trainingColors = [Rules.ColorBlack]

        if self._processPool is not None:
            print("Waiting for process pool to init")
            self._waitProcessPoolInit()

        print("Start training")
        for eIndex in range(self._parameters.Episodes):
            with self._gatherCounter:
                e = self._gatherEpisodes()

            eLen += 1
            gameCount += len(e.replays)

            with self._backPropCounter:
                for c in trainingColors:
                    expectedReturn, meanWins, meanDraws = self._trainTrajectories(e.replays, c)
                    expectedReturnsHistory[c].append(expectedReturn)
                    wins[c].append(meanWins)
                    draws[c].append(meanDraws)

            if (eIndex == self._parameters.Episodes - 1) or (eIndex % self._parameters.LogEpisodeEveryN == 0):
                print(f"iteration {eIndex} - {gameCount} games simulated")

                for c in trainingColors:
                    colorName = Rules.colorName(c)
                    meanExpectedReturn = np.mean(expectedReturnsHistory[c][-20:])
                    meanWins = np.mean(wins[c][-20:])
                    meanDraws = np.mean(draws[c][-20:])
                    print(f"    {colorName:>14}: {meanExpectedReturn:>6.2f} Mean Exp. Ret")
                    print(f"    {colorName:>14}: {meanWins:>6.2f} meanWins")
                    print(f"    {colorName:>14}: {meanDraws:>6.2f} meanDraws")

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
                wins = {
                    Rules.ColorBlack: [],
                    Rules.ColorRed: [],
                }
                draws = {
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
        _, expectedReturns, meanWins, meanDraws = self._getTrajectoriesInfos([T], color)
        expectedReturn = np.array(expectedReturns).mean()

        print(f"{Rules.colorName(color)}")
        print(f"    Expected Return: {expectedReturn}")
        for t in range(len(T)):
            ts = T[t]
            print(f"    t: {t + 1:>2d}, Return: {expectedReturns[t]:>8.2f}, Reward: {ts.reward:>5.2f}, logProb: {-ts.logProbAction:>6.3f}, c: {ts.column}, {Rules.applyName(ts.applyResult)}")

    def _trainTrajectories(self, replays, color):
        gradients, expectedReturns, meanWins, meanDraws = self._getTrajectoriesInfos(replays, color)

        if self._optimizer is not None:
            self._model.train()
            self._optimizer.zero_grad()
            gradients.mean().backward()
            self._optimizer.step()

        return expectedReturns.mean().item(), meanWins, meanDraws

    def _getTrajectoriesInfos(self, replays, color):
        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

        lenR = len(replays)
        gradients = []
        expectedReturns = np.empty(lenR, dtype=np.float32)
        meanWins = 0.0
        meanDraws = 0.0

        for ri, r in enumerate(replays):
            T = r.trajectories[color]
            lenT = len(T)
            rangeT = range(lenT)

            returnsT = np.empty(lenT, dtype=np.float32)
            futureRet = 0
            for t in reversed(rangeT):
                futureRet = T[t].reward + self._parameters.Gamma * futureRet
                returnsT[t] = futureRet

            expectedReturns[ri] = returnsT[0]
            returnsT = torch.from_numpy(returnsT)

            logProbs = [T[t].logProbAction for t in rangeT]
            logProbs = torch.cat(logProbs)

            gradient = torch.dot(-logProbs, returnsT - returnsT.mean()).view(1)
            gradients.append(gradient)

            meanWins += 1.0 if r.winColor == color else 0.0
            meanDraws += 1.0 if r.winColor == Rules.ColorNone else 0.0

        meanWins /= lenR
        meanDraws /= lenR
        gradients = torch.cat(gradients)
        return gradients, expectedReturns, meanWins, meanDraws

    def _gatherEpisodes(self):
        episode = Episode()

        if self._processPool is None:
            self._model.eval()
            replay = Trainer._localSimulate(self._model, self._opponentModel)
            episode.replays.append(replay)

            for _ in self._processRange:
                replay = Trainer._localSimulate(self._model, self._opponentModel)
                episode.replays.append(replay)
        else:
            # start simulations on external processes
            remoteReplays = self._processPool.starmap_async(Trainer._localSimulate, [(self._model, self._opponentModel) for i in self._processRange])

            # keep some work for ourselves
            replay = Trainer._localSimulate(self._model, self._opponentModel)
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

    def _localSimulate(blackModel, redModel):
        global localSimulation
        localSimulation.reset()
        return localSimulation.run(blackModel=blackModel, redModel=redModel)
