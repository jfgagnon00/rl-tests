import json
import matplotlib.pyplot as plt
import numpy as np
import os

from rules import *


if __name__ == "__main__":
    data = np.empty(1)
    i = 0
    colorName = Rules.colorName(Rules.ColorBlack)
    while True:
        filename = f"expected-return-history{i}.json"

        if not os.path.exists(filename):
            break

        print(f"Reading {filename}")
        with open(filename, "r") as f:
            readData = json.load(f)

        readData = readData["expectedReturnHistory"][colorName] # ["loss"] #
        data = np.append(data, readData)

        i += 1

    # data = data[::10]

    w = 500
    yPrime = data
    y = np.convolve(data, np.ones(w), 'same') / w
    x = range(len(y))

    # print(plt.rcParams['agg.path.chunksize'])
    # plt.rcParams['agg.path.chunksize'] = 100

    print(f"Plot graph")
    plt.title("Loss over time")
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(x, yPrime)
    plt.plot(x, y)
    # plt.ylim([0, 0.5])
    plt.show()
    print(f"Plot graph done")
