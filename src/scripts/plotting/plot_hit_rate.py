"""
This script is created to plot cache hit rate.
"""
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
import numpy as np


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
        txts_tmp = [x for x in files if x.endswith(".txt")]
        txts += txts_tmp

    txts = sorted([os.path.join(args.directory, x) for x in txts])

    colors = cm.rainbow(np.linspace(0, 1, len(txts)))

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    #fig.suptitle("Hit rate evaluation")

    markers = ['^', '*', 'X', '1', '.', 'o', '<', 'v', '2', 's', ',', 'p', 'P' '8', 'h', 'H', '+', '3', 'x', '>',
               'D', 'd', '|', '4', '_']

    plotted = []
    tmp_map = {}
    txts = ["nn_3.txt", "nn_10.txt", "nn_50.txt", "nn_200.txt", "nn_1000.txt"]
    txts = [os.path.join(args.directory, x) for x in txts]

    for i, (file_name, col) in enumerate(zip(txts, colors)):
        with open(file_name, "r") as f:
            lines = f.readlines()
            line_name = lines[0]
            lines = lines[1:]

            lines = [x.split() for x in lines]

            cache_sizes = [int(x[0]) for x in lines]
            cache_hits = [float(x[1]) for x in lines]

            for s, h in zip(cache_sizes, cache_hits):
                if i == 0:
                    tmp_map[s] = h
                elif tmp_map[s] < h:
                    tmp_map[s] = h

    for i, (file_name, col) in enumerate(zip(txts, colors)):
        with open(file_name, "r") as f:
            lines = f.readlines()
            line_name = lines[0]
            lines = lines[1:]

            lines = [x.split() for x in lines]

            cache_sizes = [int(x[0]) for x in lines]
            cache_hits = [float(x[1]) for x in lines]

            # if i == 0:
            #     tmp_map = {x[0] : x[1] for x in zip(cache_sizes, cache_hits)}

            cache_hits = [x[1] / tmp_map[x[0]] for x in zip(cache_sizes, cache_hits)]

            pl_line, = plt.plot(cache_sizes, cache_hits, c=col, label=line_name, marker=markers[i])
            plotted.append(pl_line)

    plt.xlabel("Cache size")
    plt.ylabel("Hit ratio compared to best")

    plt.legend(handles=plotted)

    plt.tight_layout()
    plt.grid(True)
    plot_name = os.path.join(args.directory, "cache_hit_plot.png")
    plt.savefig(plot_name)


if __name__ == "__main__":
    main()
