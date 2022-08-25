import numpy as np
import sys
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    infile = sys.argv[1]
    app = sys.argv[2]
    loss_name = sys.argv[3] if len(sys.argv) > 3 else "Loss"
    epoches = []
    losses = []
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 1 and tokens[0] == "Epoch:" and tokens[2] == f"{loss_name}:":
                epoches.append(float(tokens[1]))
                losses.append(float(tokens[3]))
    plt.figure()
    plt.plot(epoches, losses)
    plt.title(f"Training {loss_name} curve of {app} (on unique samples)")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training {loss_name}")
    plt.savefig(f"training_unique_{loss_name}.png", dpi=200)

    plt.figure()
    plt.plot(epoches[-2000:], losses[-2000:])
    plt.title(f"Training {loss_name} curve of {app} (on unique samples) on Tail Iters")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training {loss_name}")
    plt.savefig(f"training_unique_{loss_name}_tail.png", dpi=200)

    plt.figure()
    losses = [math.log2(x) for x in losses]
    plt.plot(epoches, losses)
    plt.title(f"Training {loss_name} curve of {app} (on unique samples)")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training {loss_name} (log2 scaled)")
    plt.savefig(f"training_unique_{loss_name}_log.png", dpi=200)

    plt.figure()
    plt.plot(epoches[-2000:], losses[-2000:])
    plt.title(f"Training {loss_name} curve of {app} (on unique samples) on Tail Iters")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training {loss_name} (log2 scaled)")
    plt.savefig(f"training_unique_{loss_name}_log_tail.png", dpi=200)