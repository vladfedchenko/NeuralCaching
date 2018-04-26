"""
This script is created to produce simple plots that visualize predicted item order compared to actual order.
"""
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_order_file",
                        type=str,
                        help="input data file with order of items")
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

    fig = plt.figure(1, figsize=(args.size_x, args.size_y))
    fig.suptitle("Item order")
    sub1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    with open(args.input_order_file, "r") as f:
        lines = [x.split() for x in f.readlines()]

        order = [int(x[0]) for x in lines]
        pred_pops = [float(x[1]) for x in lines]
        real_pops = [float(x[2]) for x in lines]

    x = range(1, len(order) + 1)

    sub1.plot(x, order, "bs", markersize=0.5)
    sub1.set_xlabel("Predicted position")
    sub1.set_ylabel("Actual position")

    sub2 = plt.subplot2grid((3, 1), (2, 0))
    axis = plt.gca()
    axis.set_yscale("log")

    pred_dots, = sub2.plot(order, pred_pops, "rs", markersize=0.5, label="Predicted")
    real_dots, = sub2.plot(order, real_pops, "gs", markersize=0.5, label="Real")

    sub2.legend(handles=[pred_dots, real_dots])
    sub2.set_xlabel("Item")
    sub2.set_ylabel("Population")

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
