"""
This script is created to evaluate the error on generated traces using feedforward neural network.
"""
import pandas as pd
import numpy as np
from neural_nets import TorchFeedforwardNN
import torch
import torch.utils.data
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
    parser.add_argument("-mb",
                        "--mini_batch",
                        type=int,
                        help="minibatch size, 1000 is default",
                        default=1000)
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
    parser.add_argument("-ha",
                        "--hidden_activation",
                        help="activation to use on hidden layers",
                        type=str)
    parser.add_argument("-oa",
                        "--out_activation",
                        help="activation to use on out layer",
                        type=str)
    parser.add_argument("-ihl",
                        "--input_has_labels",
                        help="pass this is input has class label. Needed for optimal predictor evaluation",
                        action="store_true")
    parser.add_argument("--seed",
                        help="seed for item sampling",
                        type=int)
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    data = pd.read_csv(args.input, header=None)

    if args.sample:
        data = data.sample(n=args.sample)

    n = len(data)
    train_size = n * args.train_validation_split

    train_data = data.sample(n=int(train_size), random_state=args.seed)
    valid_data = data.drop(train_data.index)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Running on: {0}".format(device))

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    if args.unpickle_file is not None:
        filename = "nn_{0}.p".format(args.unpickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
        nn = TorchFeedforwardNN(layers,
                                hidden_activation=args.hidden_activation,
                                out_activation=args.out_activation)
        if torch.cuda.is_available():
            nn.to(device)

    inp_train = torch.from_numpy(np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1]))
    outp_train = torch.from_numpy(np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]]))

    inp_valid = np.matrix(valid_data.iloc[:, 1:valid_data.shape[1] - 1])
    outp_valid = np.matrix(valid_data.iloc[:, valid_data.shape[1] - 1:valid_data.shape[1]])

    # tmp = inp_valid
    # if args.input_has_labels:
    #     tmp = tmp[:, 1:]
    #
    # tmp = np.exp(tmp) - 10 ** -5  # transform from log
    # mean_vals = np.mean(tmp, axis=1)
    # mean_vals = np.log(mean_vals + 10 ** -5)  # transform to log
    # # print(tmp[0, :], outp_valid[0, :])
    #
    # err = mean_vals - outp_valid
    # optim_err = np.mean(np.multiply(err, err))
    optim_err = 0.0

    inp_valid = torch.from_numpy(inp_valid)
    outp_valid = torch.from_numpy(outp_valid)

    if torch.cuda.is_available():
        inp_train = inp_train.to(device)
        outp_train = outp_train.to(device)
        inp_valid = inp_valid.to(device)
        outp_valid = outp_valid.to(device)

    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "a+") as f:
        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inp_train, outp_train),
                                                       batch_size=args.mini_batch,
                                                       shuffle=True)
            for inp, target in tqdm(train_loader, desc="Running minibatches"):
                nn.backpropagation_learn(inp, target, args.learning_rate, show_progress=True, stochastic=False)

            train_err = nn.evaluate(inp_train, outp_train)
            valid_err = nn.evaluate(inp_valid, outp_valid)

            f.write("{} {} {}\n".format(optim_err, train_err, valid_err))
            f.flush()

    if args.pickle_file is not None:
        filename = "nn_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
