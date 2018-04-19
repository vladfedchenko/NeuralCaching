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
    sub1 = plt.subplot(111)

    with open(args.input_order_file, "r") as f:
        order = [int(x) for x in f.readlines()]

        x = range(1, len(order) + 1)

        sub1.plot(x, order, "bs", markersize=0.5)
        sub1.set_xlabel("Predicted position")
        sub1.set_ylabel("Actual position")

    plt.savefig("{0}.png".format(args.plot_name))


if __name__ == "__main__":
    main()
