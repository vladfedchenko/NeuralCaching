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
        mean_list = [float(line[0]) for line in lines]
        deviation_list = [float(line[1]) for line in lines]
        min_list = [float(line[2]) for line in lines]
        max_list = [float(line[3]) for line in lines]

    iters = range(1, len(mean_list) + 1)
    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Feedforward NN evaluation error")

    sub1 = plt.subplot(211)
    axis = plt.gca()
    axis.set_yscale("log")
    line_mean, = sub1.plot(iters, mean_list, "b", label="Mean")
    line_min, = sub1.plot(iters, min_list, "g", label="Min")
    line_max, = sub1.plot(iters, max_list, "r", label="Max")
    sub1.legend(handles=[line_mean, line_min, line_max])
    sub1.set_xlabel("Iterations")
    sub1.set_ylabel("Error")

    y_limit = axis.get_ylim()

    sub2 = plt.subplot(212)
    axis = plt.gca()
    axis.set_yscale("log")
    axis.set_ylim(y_limit)
    line_dev, = sub2.plot(iters, deviation_list, "b", label="Standard deviation")
    sub2.legend(handles=[line_dev])
    sub2.set_xlabel("Iterations")
    sub2.set_ylabel("Standard deviation")

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
