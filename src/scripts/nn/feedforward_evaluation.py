"""
This script is created to evaluate the error on generated traces using feedforward neural network.
"""
import pandas as pd
import numpy as np
from neural_nets.feedforward_nn import FeedforwardNeuralNet, sigmoid, sigmoid_deriv
import argparse
from tqdm import tqdm
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input dataset")
    parser.add_argument("-i",
                        "--iterations",
                        type=int,
                        help="iterations to do",
                        default=1000)
    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        help="learning rate",
                        default=0.1)
    parser.add_argument("-ts",
                        "--train_sample_size",
                        type=int,
                        help="fraction of samples to use on learning step. If not passed - whole dataset is used",
                        default=None)
    parser.add_argument("-es",
                        "--eval_sample_size",
                        type=int,
                        help="fraction of samples to use on evaluation step. If not passed - whole dataset is used",
                        default=None)
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        help="file to output accuracy values",
                        default="accuracy.txt")
    parser.add_argument("-pf",
                        "--pickle_file",
                        type=str,
                        help="pickle file to dump neural network state after learning state",
                        default=None)
    parser.add_argument("-uf",
                        "--unpickle_file",
                        type=str,
                        help="pickle file to restore neural network state from at the beginning",
                        default=None)
    args = parser.parse_args()

    data = pd.read_csv(args.input, header=None)

    if args.unpickle_file is not None:
        with open(args.unpickle_file, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        nn = FeedforwardNeuralNet([data.shape[1] - 1, 20, 1], out_activ=sigmoid, out_activ_deriv=sigmoid_deriv)

    with open(args.output_file, "w") as f:
        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            if args.train_sample_size is None:
                train_data = data
            else:
                train_data = data.sample(n=args.train_sample_size)
            inp = np.matrix(train_data.iloc[:, 0:train_data.shape[1] - 1])
            outp = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])
            nn.backpropagation_learn(inp, outp, args.learning_rate, show_progress=True)

            if args.eval_sample_size is None:
                eval_data = data
            else:
                eval_data = data.sample(n=args.eval_sample_size)
            inp = np.matrix(eval_data.iloc[:, 0:eval_data.shape[1] - 1])
            outp = np.matrix(eval_data.iloc[:, eval_data.shape[1] - 1:eval_data.shape[1]])

            mean_acc, deviation, min_acc, max_acc = nn.evaluate(inp, outp, show_progress=True)
            f.write(f"{mean_acc} {deviation} {min_acc} {max_acc}\n")

        inp = np.matrix(data.iloc[:, 0:data.shape[1] - 1])
        outp = np.matrix(data.iloc[:, data.shape[1] - 1:data.shape[1]])

        mean_acc, deviation, min_acc, max_acc = nn.evaluate(inp, outp, show_progress=True)
        f.write(f"{mean_acc} {deviation} {min_acc} {max_acc}\n")

    if args.pickle_file is not None:
        with open(args.pickle_file, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
