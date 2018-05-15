"""
This script is made to plot the data produced by feedforward_evaluation.py script.
This script will produce two plots. First with mean, min and max accuracy. Second with standard deviation.
"""
import argparse
import matplotlib.pyplot as plt


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
        optim_list = [float(line[0]) for line in lines]
        train_err = [float(line[1]) for line in lines]
        valid_err = [float(line[2]) for line in lines]

    iters = range(1, len(optim_list) + 1)
    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Feedforward NN evaluation error")

    sub1 = plt.subplot(111)
    axis = plt.gca()
    axis.set_yscale("log")
    line_optim, = sub1.plot(iters, optim_list, "g", label="Optimal")
    line_min, = sub1.plot(iters, train_err, "b", label="Training error")
    line_max, = sub1.plot(iters, valid_err, "r", label="Validation error")
    sub1.legend(handles=[line_optim, line_min, line_max])
    sub1.set_xlabel("Iterations")
    sub1.set_ylabel("Error")

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
