"""
This script takes a pickled NN as an input and prints to file it's linear features and weights of its layers.
"""
import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nn_pickle",
                        type=str,
                        help="NN pickle file")
    parser.add_argument("-olf",
                        "--out_linear_features",
                        type=str,
                        help="file to output linear features")
    parser.add_argument("-ow",
                        "--out_weights",
                        type=str,
                        help="file prefix to output NN weights")
    args = parser.parse_args()

    with open(args.nn_pickle, "rb") as unpickle_file:
        nn = pickle.load(unpickle_file)

    weights = nn.get_weights()

    res = weights[0]
    for i in range(1, len(weights)):
        cur = weights[i][1:, :]
        res = np.matmul(res, cur)

    np.savetxt(args.out_linear_features, res)

    for i, matr in enumerate(weights):
        np.savetxt("{0}_{1}.txt".format(args.out_weights, i + 1), matr)


if __name__ == "__main__":
    main()
