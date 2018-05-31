"""
This script is made to plot the data produced by feedforward_evaluation.py script.
This script will produce two plots. First with mean, min and max accuracy. Second with standard deviation.
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
    parser.add_argument("-log",
                        "--log_scale",
                        help="plot errors in log scale",
                        action="store_true")
    args = parser.parse_args()

    inp = os.path.join(args.directory, "error.txt")
    with open(inp, "r") as f:
        lines = [x.split() for x in f.readlines()]
        train_err = [float(line[0]) for line in lines]
        valid_err = [float(line[1]) for line in lines]

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle(("Feedforward NN evaluation error\n"
                  "Training set average predictor error={}\n"
                  "Validation set average predictor error={}").format(train_err[0], valid_err[0]))

    aver_train_err = train_err[0]
    aver_valid_err = valid_err[0]

    train_err = train_err[1:]
    valid_err = valid_err[1:]

    iters = range(1, len(valid_err) + 1)

    # max_ = np.max(valid_err)
    # min_ = np.min(train_err)
    # diff = max_ - min_

    sub1 = plt.subplot(111)
    if args.log_scale:
        axis = plt.gca()
        axis.set_yscale("log")
    # line_optim, = sub1.plot(iters, optim_list, "g", label="Optimal")
    line_min, = sub1.plot(iters, train_err, "b", label="Training error")
    line_max, = sub1.plot(iters, valid_err, "r", label="Validation error")

    aver_train_line = sub1.axhline(aver_train_err, color="c", label="Training error (average predictor)")
    aver_valid_line = sub1.axhline(aver_valid_err, color="m", label="Validation error (average predictor)")

    # sub1.legend(handles=[line_optim, line_min, line_max])
    sub1.legend(handles=[line_min, line_max, aver_train_line, aver_valid_line])
    sub1.set_xlabel("Iterations")
    if args.log_scale:
        sub1.set_ylabel("Error (log)")
    else:
        sub1.set_ylabel("Error")
    # sub1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # plt.ylim(min_ - 0.1 * diff, max_ + 0.1 * diff)

    plot_name = os.path.join(args.directory, "err_plot.png")
    plt.savefig(plot_name)


if __name__ == "__main__":
    main()
