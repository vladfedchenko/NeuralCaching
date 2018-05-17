"""
This script is created to evaluate the error on generated traces using feedforward neural network.
"""
import pandas as pd
import numpy as np
from neural_nets import FeedforwardNeuralNet, sigmoid, sigmoid_deriv, relu, relu_deriv
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
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    data = pd.read_csv(args.input, header=None)

    if args.sample:
        data = data.sample(n=args.sample)

    n = len(data)
    train_size = n * args.train_validation_split

    train_data = data.sample(n=int(train_size))
    valid_data = data.drop(train_data.index)

    if args.unpickle_file is not None:
        filename = "nn_{0}.p".format(args.unpickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)

    else:
        act_hid = None
        act_hid_deriv = None
        act_out = None
        act_out_deriv = None

        if args.hidden_activation == "sigmoid":
            act_hid = sigmoid
            act_hid_deriv = sigmoid_deriv

        if args.out_activation == "sigmoid":
            act_out = sigmoid
            act_out_deriv = sigmoid_deriv

        if args.hidden_activation == "relu":
            act_hid = relu
            act_hid_deriv = relu_deriv

        if args.out_activation == "relu":
            act_out = relu
            act_out_deriv = relu_deriv

        layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
        nn = FeedforwardNeuralNet(layers,
                                  internal_activ=act_hid,
                                  internal_activ_deriv=act_hid_deriv,
                                  out_activ=act_out,
                                  out_activ_deriv=act_out_deriv)

    inp_train = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
    outp_train = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])

    inp_valid = np.matrix(valid_data.iloc[:, 1:valid_data.shape[1] - 1])
    outp_valid = np.matrix(valid_data.iloc[:, valid_data.shape[1] - 1:valid_data.shape[1]])

    tmp = inp_valid
    if args.input_has_labels:
        tmp = tmp[:, 1:]

    tmp = np.exp(tmp) - 10**-5
    # print(tmp[0, :], outp_valid[0, :])

    mean_vals = np.mean(tmp, axis=1)
    err = mean_vals - outp_valid
    optim_err = np.mean(np.multiply(err, err))

    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "a+") as f:
        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            nn.backpropagation_learn(inp_train, outp_train, args.learning_rate, show_progress=True, stochastic=True)

            train_err = nn.evaluate(inp_train, outp_train, show_progress=True)[0]
            valid_err = nn.evaluate(inp_valid, outp_valid, show_progress=True)[0]

            f.write("{} {} {}\n".format(optim_err, train_err, valid_err))
            f.flush()

    if args.pickle_file is not None:
        filename = "nn_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
