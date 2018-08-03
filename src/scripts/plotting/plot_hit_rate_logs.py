"""
This script is created to plot cache hit rate.
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
import numpy as np
matplotlib.use('agg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",
                        type=str,
                        help="directory with data files")
    parser.add_argument("-x",
                        "--size_x",
                        type=int,
                        help="plot size in x axis",
                        default=10)
    parser.add_argument("-y",
                        "--size_y",
                        type=int,
                        help="plot size in y axis",
                        default=10)
    args = parser.parse_args()

    txts = []
    for root, dirs, files in os.walk(args.directory):
        txts_tmp = [x for x in files if x.endswith(".log")]
        txts += txts_tmp

    txts = sorted([os.path.join(args.directory, x) for x in txts])

    colors = cm.gnuplot(np.linspace(0, 1, len(txts)))

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Hit rate evaluation")

    plotted = []
    for file_name, col in zip(txts, colors):
        with open(file_name, "r") as f:
            lines = f.readlines()
            line_name = lines[0]
            lines = lines[1:]

            lines = [x.split() for x in lines]

            cache_sizes = [int(x[1]) for x in lines]
            cache_hits = [float(x[2]) for x in lines]

            pl_line, = plt.plot(cache_sizes, cache_hits, c=col, label=line_name)
            plotted.append(pl_line)

    plt.xlabel("Requests processed")
    plt.ylabel("Hit ratio")

    plt.legend(handles=plotted)

    plot_name = os.path.join(args.directory, "cache_hit_plot.png")
    plt.plot()
    plt.savefig(plot_name)


if __name__ == "__main__":
    main()
