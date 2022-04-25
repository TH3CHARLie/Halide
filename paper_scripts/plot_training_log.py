import numpy as np
import sys
import matplotlib.pyplot as plt
import math

INTERVAL = 5000

if __name__ == "__main__":
    infile = sys.argv[1]
    app_name = sys.argv[2]
    epoches = []
    losses = []
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 1 and tokens[0] == "Epoch:":
                epoches.append(float(tokens[1]))
                losses.append(float(tokens[3]))
    plt.figure()
    plt.plot(epoches[::INTERVAL], losses[::INTERVAL])
    plt.title(f"Training Loss curve of {app_name} (on unique samples)")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig(f"training_unique_loss_{app_name}.png", dpi=200)

    plt.figure()
    losses = [math.log2(x) for x in losses]
    plt.plot(epoches, losses)
    plt.title(f"Training Loss curve of {app_name} (on unique samples)")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (log2 scaled)")
    plt.savefig(f"training_unique_loss_log_{app_name}.png", dpi=200)
