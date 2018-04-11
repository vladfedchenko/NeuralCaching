"""
This script is made to plot the data produced by plot_distance_to_distribution.py.
This script will produce two plots. First with distance to distribution metric over iterations.
Second with artificial cache hit metric.
"""
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_distance",
                        type=str,
                        help="input data file with distance to distribution")
    parser.add_argument("input_cache",
                        type=str,
                        help="input data file with cache hit")
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

    with open(args.input_distance, "r") as f:
        distance = [float(x) for x in f.readlines()]

    iters = range(1, len(distance) + 1)
    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Feedforward NN evaluation")

    sub1 = plt.subplot(211)
    sub1.plot(iters, distance, "b")
    sub1.set_xlabel("Iterations")
    sub1.set_ylabel("Distance to distribution")

    with open(args.input_cache, "r") as f:
        lines = [x.split() for x in f.readlines()]
        best_hit = [float(line[0]) for line in lines]
        pred_hit = [float(line[1]) for line in lines]

    sub2 = plt.subplot(212)
    best_line, = sub1.plot(range(1, len(best_hit) + 1), best_hit, "g", label="Optimal")
    pred_min, = sub1.plot(range(1, len(pred_hit) + 1), pred_hit, "r", label="Predicted")

    sub2.legend(handles=[best_line, pred_min])
    sub2.set_xlabel("Cache size")
    sub2.set_ylabel("Hit rate")

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
