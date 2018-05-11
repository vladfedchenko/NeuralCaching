"""
This script is created for plotting the data created by "observe_order_dynamics.py" script.
"""
import argparse
import matplotlib.pyplot as plt
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_prefix",
                        type=str,
                        help="plot file name prefix")
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
                        default=15)
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.directory):
        data_files = sorted([f for f in files if f.endswith(".txt")])
        for file in data_files:
            print(file)
            num = int(re.findall(r"\d+", file)[1])
            filename = os.path.join(root, file)

            with open(filename, "r") as f:
                lines = [x.split() for x in f.readlines()]

                order = [int(x[0]) for x in lines]
                pred_pops = [float(x[1]) for x in lines]
                real_pops = [float(x[2]) for x in lines]

            tmp = [x[0] for x in
                   sorted(sorted(zip(order, real_pops), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)]
            order_map = {x[1]: x[0] for x in zip(range(1, len(tmp) + 1), tmp)}

            order_by_pop = [order_map[x] for x in order]

            x = range(1, len(order) + 1)

            pred_pops_pos = [x[0] for x in zip(pred_pops, order) if x[0] >= 0.0]
            order_pos = [x[1] for x in zip(pred_pops, order) if x[0] >= 0.0]

            pred_pops_neg = [abs(x[0]) for x in zip(pred_pops, order) if x[0] < 0.0]
            order_neg = [x[1] for x in zip(pred_pops, order) if x[0] < 0.0]

            fig = plt.figure(1, figsize=(args.size_x, args.size_y))
            fig.suptitle("Item order")
            sub1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

            sub1.plot(order_by_pop, x, "bs", markersize=0.5)
            sub1.set_xlabel("Actual position")
            sub1.set_ylabel("Predicted position")

            sub2 = plt.subplot2grid((3, 1), (2, 0))
            axis = plt.gca()
            axis.set_yscale("log")

            real_dots, = sub2.plot(order, real_pops, "gs", markersize=0.5, label="Real")
            pred_dots_neg, = sub2.plot(order_neg, pred_pops_neg, "rs", markersize=0.5,
                                       label="Predicted (negative, abs)")
            pred_dots_pos, = sub2.plot(order_pos, pred_pops_pos, "bs", markersize=0.5, label="Predicted (positive)")

            sub2.legend(handles=[pred_dots_pos, pred_dots_neg, real_dots])
            sub2.set_xlabel("Item")
            sub2.set_ylabel("Popularity")

            plt.savefig("{0}{1}{2}.png".format(os.path.join(args.directory, args.plot_prefix),
                                               "_order_plot_",
                                               num))


if __name__ == "__main__":
    main()
