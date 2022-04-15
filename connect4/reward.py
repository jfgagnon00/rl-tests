from rules import *

class Reward:
    def Get(applyResult):
        return {
            Rules.ApplyInvalid: -15.0,
            Rules.ApplyInconclusive: -0.1,
            Rules.ApplyTie: 0.0,
            Rules.ApplyWonVertical: 15.0,
            Rules.ApplyWonHorizontal: 15.0,
            Rules.ApplyWonDiag1: 15.0,
            Rules.ApplyWonDiag2: 15.0,
        }[applyResult]

    def Return(trajectory, t, gamma):
        r = 0
        gammaPrime = 1
        for tprime in range(t, len(trajectory)):
            r += gammaPrime * trajectory[tprime].reward
            gammaPrime *= gamma
        return r