import json
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    expectedReturnHistory = np.empty(1)
    i = 0
    while True:
        filename = f"expected-return-history{i}.json"

        if not os.path.exists(filename):
            break

        print(f"Reading {filename}")
        with open(filename, "r") as f:
            data = json.load(f)

        data = data["expectedReturnHistory"]
        expectedReturnHistory = np.append(expectedReturnHistory, data)

        i += 1

    w = 1000
    yPrime = expectedReturnHistory
    y = np.convolve(expectedReturnHistory, np.ones(w), 'same') / w
    x = range(len(y))

    print(f"Plot graph")
    plt.xlabel('Episodes')
    plt.ylabel('Expected Return')
    # plt.plot(x, yPrime)
    plt.plot(x, y)
    plt.ylim([0, 0.1])
    plt.show()
    print(f"Plot graph done")
