"""
This script is made to plot the data produced by plot_distance_to_distribution.py.
This script will produce two plots. First with distance to distribution metric over iterations.
Second with artificial cache hit metric.
"""
import argparse
import matplotlib.pyplot as plt
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
    parser.add_argument("-xo",
                        "--size_xo",
                        type=int,
                        help="order plot size in x axis",
                        default=10)
    parser.add_argument("-yo",
                        "--size_yo",
                        type=int,
                        help="order plot size in y axis",
                        default=15)
    args = parser.parse_args()

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Feedforward NN evaluation")

    dist_file = None
    cache_file = None
    order_file = None

    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.endswith("distance.txt"):
                dist_file = os.path.join(root, file)
            if file.endswith("cache_hit.txt"):
                cache_file = os.path.join(root, file)
            if file.endswith("order.txt"):
                order_file = os.path.join(root, file)

    if dist_file is not None:
        with open(dist_file, "r") as f:
            lines = [x.split() for x in f.readlines()]
            distance = [float(x[0]) for x in lines]
            errors = [float(x[1]) for x in lines]

        iters = range(1, len(distance) + 1)

        sub1 = plt.subplot2grid((2, 2), (0, 0))
        sub1.plot(iters, distance, "b")
        sub1.set_xlabel("Iterations")
        sub1.set_ylabel("Distance to distribution")

        sub3 = plt.subplot2grid((2, 2), (0, 1))
        sub3.plot(iters, errors, "b")
        sub3.set_xlabel("Iterations")
        sub3.set_ylabel("Error")

    if cache_file is not None:
        with open(cache_file, "r") as f:
            lines = [x.split() for x in f.readlines()]
            best_hit = [float(line[0]) for line in lines]
            pred_hit = [float(line[1]) for line in lines]

        sub2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        best_line, = sub2.plot(range(1, len(best_hit) + 1), best_hit, "g", label="Optimal")
        pred_min, = sub2.plot(range(1, len(pred_hit) + 1), pred_hit, "r", label="Predicted")

        sub2.legend(handles=[best_line, pred_min])
        sub2.set_xlabel("Cache size")
        sub2.set_ylabel("Hit rate")

    plt.savefig(os.path.join(args.directory, "dist_plot.png"))

    if order_file is not None:
        with open(order_file, "r") as f:
            lines = [x.split() for x in f.readlines()]

            order = [int(x[0]) for x in lines]
            pred_pops = [float(x[1]) for x in lines]
            real_pops = [float(x[2]) for x in lines]

        tmp = [x[0] for x in sorted(sorted(zip(order, real_pops), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)]
        order_map = {x[1]: x[0] for x in zip(range(1, len(tmp) + 1), tmp)}

        order_by_pop = [order_map[x] for x in order]

        x = range(1, len(order) + 1)

        pred_pops_pos = [x[0] for x in zip(pred_pops, order) if x[0] >= 0.0]
        order_pos = [x[1] for x in zip(pred_pops, order) if x[0] >= 0.0]

        pred_pops_neg = [abs(x[0]) for x in zip(pred_pops, order) if x[0] < 0.0]
        order_neg = [x[1] for x in zip(pred_pops, order) if x[0] < 0.0]

        fig = plt.figure(2, figsize=(args.size_xo, args.size_yo))
        fig.suptitle("Item order")
        sub1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        sub1.plot(order_by_pop, x, "bs", markersize=0.5)
        sub1.set_xlabel("Actual position")
        sub1.set_ylabel("Predicted position")

        sub2 = plt.subplot2grid((3, 1), (2, 0))
        axis = plt.gca()
        axis.set_yscale("log")

        pred_dots_neg, = sub2.plot(order_neg, pred_pops_neg, "rs", markersize=0.5, label="Predicted (negative, abs)")
        pred_dots_pos, = sub2.plot(order_pos, pred_pops_pos, "bs", markersize=0.5, label="Predicted (positive)")
        real_dots, = sub2.plot(order, real_pops, "gs", markersize=0.5, label="Real")

        sub2.legend(handles=[pred_dots_pos, pred_dots_neg, real_dots])
        sub2.set_xlabel("Item")
        sub2.set_ylabel("Popularity")

        plt.savefig(os.path.join(args.directory, "order_plot.png"))


if __name__ == "__main__":
    main()
