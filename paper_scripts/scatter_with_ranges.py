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

def plot(predictions_file, app, output_dir, lower_bound, upper_bound, name):
  predicted_label = name + " Predicted"
  actual_label = "Actual"
  data = pd.read_csv(predictions_file, names=[predicted_label, actual_label])

  for i, item in enumerate(data[predicted_label]):
    data[predicted_label][i] = 1.0 / item
  for i, item in enumerate(data[actual_label]):
    data[actual_label][i] = 1.0 / item


  r2 = rsquared(data, predicted_label, actual_label)
  title = "{}: Throughput Predictions ($R^2$ = {:.2f}; Loss = {:.2f})".format(app, r2, relative_loss(data, predicted_label, actual_label))
  fig, ax = plt.subplots()
  plt.scatter(x=predicted_label, y=actual_label, data=data, s=5, linewidth=0.05, alpha=0.5)

  lower_bound_label = "Lower Bound"
  upper_bound_label = "Upper Bound"
  lower_bound_data = pd.read_csv(lower_bound, names=[lower_bound_label, actual_label])
  upper_bound_data = pd.read_csv(upper_bound, names=[upper_bound_label, actual_label])
  lower_bound_data[lower_bound_label] = 1.0 / lower_bound_data[lower_bound_label]
  upper_bound_data[upper_bound_label] = 1.0 / upper_bound_data[upper_bound_label]

  # print(data[predicted_label])
  # print(lower_bound_data[lower_bound_label])
  plt.vlines(x=data[predicted_label], ymin=lower_bound_data[lower_bound_label], ymax=upper_bound_data[upper_bound_label], colors='red', lw=0.5)


  ax.set_title(title)
  # plt.xscale('log', base=2)
  # plt.yscale('log', base=2)

  max = np.ceil(np.max(np.max(data)))
  min = np.floor(np.min(np.min(data)))

  # plt.plot([0, max], [0, max], linewidth=1)
  ax.grid(True, alpha=0.4, linestyle='--')
  ax.grid(True, which='minor', alpha=0.4, linestyle='--')

  plt.xlabel(predicted_label)
  plt.ylabel(actual_label)

  for axis in [ax.get_xaxis(), ax.get_yaxis()]:
    axis.set_major_formatter(FormatStrFormatter("%.7f"))
    axis.set_minor_formatter(ScalarFormatter())

    for tick in axis.get_minor_ticks():
      tick.label.set_fontsize(5)

    for tick in axis.get_major_ticks():
      tick.label.set_fontsize(7)

  filename = "{}/{}.png".format(output_dir, name + "_" + app)
  plt.savefig(filename, dpi=200)
  print("Saved scatter plot to {}".format(filename))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--app", type=str, required=True)
  parser.add_argument("--output", type=str, required=True)
  parser.add_argument("--lower", type=str, required=True)
  parser.add_argument("--upper", type=str, required=True)

  args = parser.parse_args()
  plot(args.lower, args.app, args.output, args.lower, args.upper, "Lower_bound")
  plot(args.upper, args.app, args.output, args.lower, args.upper, "Upper_bound")
