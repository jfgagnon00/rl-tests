from rules import *

class Reward:
    def Get(applyResult):
        return {
            Rules.ApplyInvalid: -10.0,
            Rules.ApplyInconclusive: 0.0,
            Rules.ApplyTie: 0.5,
            Rules.ApplyWonVertical: 1.0,
            Rules.ApplyWonHorizontal: 1.0,
            Rules.ApplyWonDiag1: 1.0,
            Rules.ApplyWonDiag2: 1.0,
        }[applyResult]
