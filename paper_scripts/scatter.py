from scipy import stats
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from rsquared import rsquared, relative_loss

def plot(predictions_file, app, output_dir, plot_throughput):
  predicted_label = "Predicted "
  actual_label = "Actual "
  data = pd.read_csv(predictions_file, names=[predicted_label, actual_label])
  if plot_throughput:
    data[predicted_label] = np.min(data[predicted_label]) / data[predicted_label]
    data[actual_label] = np.min(data[actual_label])  / data[actual_label]
  

  r2 = rsquared(data, predicted_label, actual_label)
  token = "Relative Throughput" if plot_throughput else "Runtime"
  title = "{}: \n {} Predictions ($R^2$ = {:.2f}; Loss = {:.2f})".format(app, token, r2, relative_loss(data, predicted_label, actual_label))
  fig, ax = plt.subplots()
  plt.scatter(x=predicted_label, y=actual_label, data=data, s=5, linewidth=0.05, alpha=0.5)

  ax.set_title(title)


  max = np.ceil(np.max(np.max(data)))
  min = np.floor(np.min(np.min(data)))

  plt.plot([0, max], [0, max], linewidth=1)
  ax.grid(True, alpha=0.4, linestyle='--')
  ax.grid(True, which='minor', alpha=0.4, linestyle='--')

  plt.xlabel(predicted_label)
  plt.ylabel(actual_label)

  for axis in [ax.get_xaxis(), ax.get_yaxis()]:
    axis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.set_minor_formatter(ScalarFormatter())

    for tick in axis.get_minor_ticks():
      tick.label.set_fontsize(5)

    for tick in axis.get_major_ticks():
      tick.label.set_fontsize(7)

  filename = "{}/{}.png".format(output_dir, app)
  plt.savefig(filename, dpi=200)
  print("Saved scatter plot to {}".format(filename))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--predictions", type=str, required=True)
  parser.add_argument("--app", type=str, required=True)
  parser.add_argument("--output", type=str, required=True)
  parser.add_argument("--throughput", action="store_true")

  args = parser.parse_args()
  plot(args.predictions, args.app, args.output, args.throughput)
