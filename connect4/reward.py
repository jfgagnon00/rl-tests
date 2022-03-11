from rules import *

def Reward(applyResult):
    return {
        Rules.ApplyInvalid: -15,
        Rules.ApplyInconclusive: -0.1,
        Rules.ApplyTie: 1.0,
        Rules.ApplyWonVertical: 15.0,
        Rules.ApplyWonHorizontal: 15.0,
        Rules.ApplyWonDiag1: 15.0,
        Rules.ApplyWonDiag2: 15.0,
    }[applyResult]

def Return(trajectory, t, gamma):
    r = 0
    for tprime in range(t, len(trajectory)):
        scale = gamma ** (tprime - t)

        # column, logProbAction, reward, applyResult
        action, logProbAction, reward, state = trajectory[tprime]

        r += scale * reward