"""
This script is made to plot the data produced by feedforward_eval_multiple_pop.py script.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input data file")
    parser.add_argument("plot_name",
                        type=str,
                        help="plot file name")
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

    with open(args.input, "r") as f:
        lines = [x.split() for x in f.readlines()]
        sizes = [int(line[0]) for line in lines]
        optim_list = [float(line[1]) for line in lines]
        train_err = [float(line[2]) for line in lines]
        valid_err = [float(line[3]) for line in lines]

    # iters = range(1, len(optim_list) + 1)
    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Feedforward NN evaluation error")

    # max_ = np.max(valid_err)
    # min_ = np.min(train_err)
    # diff = max_ - min_

    sub1 = plt.subplot(111)
    axis = plt.gca()
    axis.set_yscale("log")
    line_optim, = sub1.plot(sizes, optim_list, "g", label="Optimal")
    line_min, = sub1.plot(sizes, train_err, "b", label="Training error")
    line_max, = sub1.plot(sizes, valid_err, "r", label="Validation error")
    sub1.legend(handles=[line_optim, line_min, line_max])
    # sub1.legend(handles=[line_min, line_max])
    sub1.set_xlabel("Population size")
    sub1.set_ylabel("Error (log)")
    # sub1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # plt.ylim(min_ - 0.1 * diff, max_ + 0.1 * diff)

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
