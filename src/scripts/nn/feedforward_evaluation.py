"""
This script is created to evaluate the error on generated traces using feedforward neural network.
"""
import pandas as pd
import numpy as np
from neural_nets.feedforward_nn import FeedforwardNeuralNet
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input dataset")
    parser.add_argument("-i",
                        "--iterations",
                        type=int,
                        help="iterations to do",
                        default=100)
    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        help="learning rate",
                        default=0.1)
    # parser.add_argument("-f",
    #                     "--fraction_learn",
    #                     type=float,
    #                     help="fraction of samples to use on learning step. Rest is used to evaluate",
    #                     default=0.9)
    args = parser.parse_args()

    inp = pd.read_csv(args.input).sample(frac=.01)
    outp = np.matrix(inp.iloc[:, inp.shape[1] - 1:inp.shape[1]])
    inp = np.matrix(inp.iloc[:, 1:inp.shape[1] - 1])
    print(inp.shape)
    print(outp.shape)

    nn = FeedforwardNeuralNet([4, 10, 1])

    for _ in range(args.iterations):
        nn.backpropagation_learn(inp, outp, 1, args.learning_rate)
        print(nn.evaluate(inp, outp))


if __name__ == "__main__":
    main()
