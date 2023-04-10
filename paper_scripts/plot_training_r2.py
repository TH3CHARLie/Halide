import sys
import matplotlib.pyplot as plt
import math

"""
plot training R^2 using excluded pipeline training logs
"""

if __name__ == "__main__":
    infile = sys.argv[1]
    app = sys.argv[2]
    epoches = []
    losses = []
    r2_train = []
    r2_validation = []
    cnt = 0
    r2_dropped = []
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 1 and tokens[0] == "Epoch:":
                epoches.append(float(tokens[1]))
                losses.append(float(tokens[3]))
                r2_train.append(float(tokens[5]))
                r2_validation.append(float(tokens[6]))
            if len(tokens) > 1 and tokens[0] == "excluding" and tokens[1] == "pipeline":
                cnt += 1
                r2_dropped.append(float(tokens[11]))
    plt.figure()
    plt.plot(epoches, losses)
    plt.title(f"Training Loss curve of {app}")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training Loss")
    plt.savefig(f"training_loss.png", dpi=200)

    plt.figure()
    plt.plot(epoches[-2000:], losses[-2000:])
    plt.title(f"Training Loss curve of {app} on Tail 2000 epochs")
    plt.xlabel("Epoch")
    plt.ylabel(f"Training Loss")
    plt.savefig(f"training_loss_tail.png", dpi=200)

    plt.figure()
    plt.plot(epoches, r2_train)
    plt.title(f"R^2 curve of Training Set")
    plt.xlabel("Epoch")
    plt.ylabel(f"R^2")
    plt.savefig(f"training_r2.png", dpi=200)

    plt.figure()
    plt.plot(epoches, r2_validation)
    plt.title(f"R^2 curve of Validation Set")
    plt.xlabel("Epoch")
    plt.ylabel(f"R^2")
    plt.savefig(f"training_r2_validation.png", dpi=200)

    print(r2_dropped)
    plt.figure()
    plt.plot(list(range(100, cnt + 100)), r2_dropped)
    plt.title(f"R^2 curve of Dropped Pipelines")
    plt.xlabel("Epoch")
    plt.ylabel(f"R^2")
    plt.savefig(f"training_r2_dropped.png", dpi=200)


