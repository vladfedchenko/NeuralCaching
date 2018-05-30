"""
This script is made to plot the data produced by evaluate_split_error.py script.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


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

    filename = os.path.join(args.directory, "error.txt")
    with open(filename, "r") as f:
        lines = [x.split() for x in f.readlines()]
        aver_predictor_err_dif = [float(line[0]) - float(line[1]) for line in lines]
        nn_err_dif = [float(line[2]) - float(line[3]) for line in lines]
        diff_list = list(sorted(zip(aver_predictor_err_dif, nn_err_dif), key=lambda x: x[0]))

    x = range(1, len(diff_list) + 1)

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Train/validation set error comparison")

    aver_pred_line, = plt.plot(x, [x[0] for x in diff_list], "g", label="Average predictor")
    nn_line, = plt.plot(x, [x[1] for x in diff_list], "r", label="NN")

    plt.legend(handles=[aver_pred_line, nn_line])

    plt.ylabel("Training error - validation error")

    plt.axhline(0.0)

    plot_name = os.path.join(args.directory, "split_error_plot.png")
    plt.savefig(plot_name)


if __name__ == "__main__":
    main()