import json
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    totalRewardHistory = np.empty(1)
    i = 0
    while True:
        filename = f"total-history{i}.json"

        if not os.path.exists(filename):
            break

        print(f"Reading {filename}")
        with open(filename, "r") as f:
            data = json.load(f)

        data = data["totalRewardHistory"]
        totalRewardHistory = np.append(totalRewardHistory, data)

        i += 1

    w = 2000
    yPrime = totalRewardHistory
    y = np.convolve(totalRewardHistory, np.ones(w), 'same') / w
    x = range(len(y))

    print(f"Plot graph")
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    # plt.plot(x, yPrime)
    plt.plot(x, y)
    plt.ylim([-1, 1])
    plt.show()
    print(f"Plot graph done")
