import multiprocessing
import torch

from joblib import Parallel, delayed
from reward import *
from rules import *
from simulation import *
from time import perf_counter


class Trainer:
    _NormalizeGradients = False
    _EpisodeScore = 10
    _TrainingColor = Rules.ColorBlack

    def __init__(self, rules, board, model, learningRate = 1e-3, gamma = 0.8):
        self._gamma = gamma
        self._rules = rules
        self._board = board
        self._model = model
        self._optimizer = torch.optim.RMSprop(model.parameters(), lr=learningRate)
        self._episodeDt = 0.0
        self._backPropDt = 0.0
        self._evalModelDt = 0.0
        self._applyRulesDt = 0.0
        self._choiceDt = 0.0
        self._trajectoryDt = 0.0

    def train(self, episodes=1000):
        print("Start new training")

        # one entry per episode
        self.expectedReturnHistory = []
        self._model.train()

        for e in range(episodes):
            episode = self._gatherEpisode(Trainer._EpisodeScore)
            winFactor = 100.0 / (episode[Rules.ColorRed] + episode[Rules.ColorBlack])

            for c in [Rules.ColorBlack, Rules.ColorRed]:
                expectedReturn = self._trainTrajectories(episode["simulations"], c)
                self.expectedReturnHistory.append(expectedReturn)

                if e % 100 == 0:
                    colorName = Rules.colorName(c)
                    ratioWins = int(episode[c] * winFactor)
                    print(f"e: {e:>5d} {colorName:5} - Expected Ret: {expectedReturn:>8.2f}, Ratio Wins: {ratioWins:>3d}%")

            if e % 100 == 0:
                print(f"    Episode        : {self._episodeDt * 1000.0:>4.2f}ms")
                print(f"        Eval Model : {self._evalModelDt * 1000.0:>4.2f}ms")
                print(f"        Choice     : {self._choiceDt * 1000.0:>4.2f}ms")
                print(f"        Apply Rules: {self._applyRulesDt * 1000.0:>4.2f}ms")
                print(f"        Trajectory : {self._trajectoryDt * 1000.0:>4.2f}ms")
                print(f"    Backprop       : {self._backPropDt * 1000.0:>4.2f}ms")
                self._episodeDt = 0.0
                self._backPropDt = 0.0
                self._evalModelDt = 0.0
                self._choiceDt = 0.0
                self._applyRulesDt = 0.0
                self._trajectoryDt = 0.0
                print()

    def debugReturns(self):
        episode = self._gatherEpisodeInfo(1)
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

    def _trainTrajectories(self, simulations, color):
        backPropStart = perf_counter()

        gradients, expectedReturns = self._getTrajectoriesInfos(simulations, color)

        if self._optimizer is not None:
            self._optimizer.zero_grad()
            gradients = torch.tensor(gradients, requires_grad=True)
            gradients.mean().backward()
            self._optimizer.step()

        backPropStop = perf_counter()
        self._backPropDt += backPropStop - backPropStart

        return np.array(expectedReturns).mean()

    def _getTrajectoriesInfos(self, simulations, color):
        # book equations are unreadable
        # https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        gradients = []
        expectedReturns = []

        for s in simulations:
            T = s.trajectories[color]
            rangeT = range(len(T))

            returnsT = [Reward.Return(T, t, self._gamma) for t in rangeT]
            returnT = np.array(returnsT).sum()
            expectedReturns.append(returnT)

            gradientsT = [-T[t].logProbAction * returnsT[t] for t in rangeT]
            gradientsT = np.array(gradientsT)

            # normalize gradients
            if Trainer._NormalizeGradients:
                gradientsT = (gradientsT - gradientsT.mean()) / (gradientsT.std() + 1e-9)

            gradientT = gradientsT.sum()
            gradients.append(gradientT)

        return gradients, expectedReturns

    def _performSimulation(self):
        board = Board(self._board.width, self._board.height)
        simulation = Simulation(self._rules, board, self._model)
        simulation.reset()
        simulation.run()
        return simulation

    def _gatherEpisode(self, winTreshold):
        episodeStart = perf_counter()

        episode = {
            "simulations": [],

            # number of wins for each color
            # episode is complete when any color wins >= winTreshold
            Rules.ColorBlack: 0,
            Rules.ColorRed: 0,
            Rules.ColorNone: 0,
        }

        cpuCount = multiprocessing.cpu_count()
        while True:
            if False:
                simulations = Parallel(n_jobs=cpuCount)(delayed(self._performSimulation)() for i in range(cpuCount))
                episode["simulations"].extend(simulations)

                done = False
                for s in simulations:
                    episode[s.winColor] += 1
                    if episode[s.winColor] >= winTreshold:
                        done = True
                    self._evalModelDt += s.evalModelDt
                    self._choiceDt += s.choiceDt
                    self._applyRulesDt += s.applyRulesDt
                    self._trajectoryDt += s.trajectoryDt

                if done:
                    break
            else:
                simulation = Simulation(self._rules, self._board, self._model)
                simulation.reset()
                simulation.run()

                self._evalModelDt += simulation.evalModelDt
                self._choiceDt += simulation.choiceDt
                self._applyRulesDt += simulation.applyRulesDt
                self._trajectoryDt += simulation.trajectoryDt

                episode["simulations"].append(simulation)
                episode[simulation.winColor] += 1
                if episode[simulation.winColor] >= winTreshold:
                    break

        episodeStop = perf_counter()
        self._episodeDt += (episodeStop - episodeStart)

        return episode