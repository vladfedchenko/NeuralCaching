"""
This script is created to evaluate the error on generated traces using feedforward neural network on population of
different sizes.
"""
import pandas as pd
import numpy as np
from neural_nets import TorchFeedforwardNN
import torch
import argparse
from tqdm import tqdm
import pickle
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input dataset")
    parser.add_argument("directory",
                        type=str,
                        help="directory to store data files")
    parser.add_argument("start_pop",
                        type=int,
                        help="starting population size")
    parser.add_argument("end_pop",
                        type=int,
                        help="end population size")
    parser.add_argument("pop_step",
                        type=int,
                        help="step in population size")
    parser.add_argument("-i",
                        "--iterations",
                        type=int,
                        help="iterations to do",
                        default=1000)
    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        help="learning rate",
                        default=0.01)
    parser.add_argument("-s",
                        "--sample",
                        type=int,
                        help="number of samples to use from dataset. If not passed - whole dataset is used",
                        default=None)
    parser.add_argument("-tvs",
                        "--train_validation_split",
                        type=float,
                        help="train - validation split fraction",
                        default=0.8)
    parser.add_argument("-pf",
                        "--pickle_file",
                        type=int,
                        help="pickle file index to dump neural network state after learning",
                        default=None)
    parser.add_argument("-uf",
                        "--unpickle_file",
                        type=int,
                        help="pickle file index to restore neural network state from at the beginning",
                        default=None)
    parser.add_argument("-ml",
                        "--middle_layers",
                        type=int,
                        help="number of middle layers",
                        default=20)
    parser.add_argument("-mln",
                        "--middle_layer_neurons",
                        type=int,
                        help="middle layers neuron count",
                        default=2)
    parser.add_argument("-shl",
                        "--sigmoid_hidden_layers",
                        help="use sigmoid on hidden layers",
                        action="store_true")
    parser.add_argument("-sol",
                        "--sigmoid_output_layers",
                        help="use sigmoid on output layer",
                        action="store_true")
    parser.add_argument("-ihl",
                        "--input_has_labels",
                        help="pass this is input has class label. Needed for optimal predictor evaluation",
                        action="store_true")
    parser.add_argument("-t",
                        "--torch",
                        help="use torch implementation of NN",
                        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    data = pd.read_csv(args.input, header=None)
    if args.sample:
        data = data.sample(n=args.sample)

    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "a+") as f:
        for pop_size in tqdm(list(reversed(range(args.start_pop, args.end_pop + 1, args.pop_step))),
                             desc="Running populations"):
            data_pop = data[data.ix[:, 0] <= pop_size]

            n = len(data_pop)
            train_size = n * args.train_validation_split

            train_data = data_pop.sample(n=int(train_size))
            valid_data = data_pop.drop(train_data.index)

            if args.unpickle_file is not None:
                filename = "nn_{}_{}.p".format(pop_size, args.unpickle_file)
                filename = os.path.join(args.directory, filename)
                with open(filename, "rb") as unpickle_file:
                    nn = pickle.load(unpickle_file)

            else:
                layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
                nn = TorchFeedforwardNN(layers,
                                        use_sigmoid_hidden=args.sigmoid_hidden_layers,
                                        use_sigmoid_out=args.sigmoid_output_layers)

            inp_train = torch.from_numpy(np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1]))
            outp_train = torch.from_numpy(np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]]))

            inp_valid = np.matrix(valid_data.iloc[:, 1:valid_data.shape[1] - 1])
            outp_valid = np.matrix(valid_data.iloc[:, valid_data.shape[1] - 1:valid_data.shape[1]])

            tmp = inp_valid
            if args.input_has_labels:
                tmp = tmp[:, 1:]

            tmp = np.exp(tmp) - 10 ** -5
            # print(tmp[0, :], outp_valid[0, :])

            mean_vals = np.mean(tmp, axis=1)
            err = mean_vals - outp_valid
            optim_err = np.mean(np.multiply(err, err))

            inp_valid = torch.from_numpy(inp_valid)
            outp_valid = torch.from_numpy(outp_valid)

            for _ in tqdm(range(args.iterations), desc="Running iterations"):
                nn.backpropagation_learn(inp_train, outp_train, args.learning_rate, show_progress=True, stochastic=True)

            train_err = nn.evaluate(inp_train, outp_train, show_progress=True)[0]
            valid_err = nn.evaluate(inp_valid, outp_valid, show_progress=True)[0]

            f.write("{} {} {} {}\n".format(pop_size, optim_err, train_err, valid_err))
            f.flush()


if __name__ == "__main__":
    main()
