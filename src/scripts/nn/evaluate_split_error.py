import pandas as pd
import numpy as np
from neural_nets import TorchFeedforwardNN
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import os


def calc_aver_error(inp, outp, has_labels):
    tmp = inp
    if has_labels:
        tmp = tmp[:, 1:]

    tmp = np.exp(tmp) - 10 ** -5  # transform from log
    mean_vals = np.mean(tmp, axis=1)
    mean_vals = np.log(mean_vals + 10 ** -5)  # transform to log
    # print(tmp[0, :], outp_valid[0, :])

    err = mean_vals - outp
    optim_err = np.mean(np.multiply(err, err))
    return optim_err


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
    parser.add_argument("-ti",
                        "--train_iterations",
                        type=int,
                        help="iterations to train NN",
                        default=10)
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
    parser.add_argument("-fc",
                        "--force_cpu",
                        help="force cpu execution for PyTorch",
                        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    print("Running on: {0}".format(device))

    data_full = pd.read_csv(args.input, header=None)

    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "w") as f:
        for seed in tqdm(range(args.iterations), desc="Running iterations"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            data = data_full
            if args.sample:
                data = data.sample(n=args.sample)

            n = len(data)
            train_size = n * args.train_validation_split

            train_data = data.sample(n=int(train_size))
            valid_data = data.drop(train_data.index)

            layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
            nn = TorchFeedforwardNN(layers,
                                    hidden_activation=args.hidden_activation,
                                    out_activation=args.out_activation)
            if torch.cuda.is_available():
                nn.to(device)

            inp_train = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
            outp_train = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])
            inp_valid = np.matrix(valid_data.iloc[:, 1:valid_data.shape[1] - 1])
            outp_valid = np.matrix(valid_data.iloc[:, valid_data.shape[1] - 1:valid_data.shape[1]])

            optim_err = calc_aver_error(inp_valid, outp_valid, args.input_has_labels)
            optim_err_train = calc_aver_error(inp_train, outp_train, args.input_has_labels)

            inp_train = torch.from_numpy(inp_train)
            outp_train = torch.from_numpy(outp_train)
            inp_valid = torch.from_numpy(inp_valid)
            outp_valid = torch.from_numpy(outp_valid)

            if torch.cuda.is_available():
                inp_train = inp_train.to(device)
                outp_train = outp_train.to(device)
                inp_valid = inp_valid.to(device)
                outp_valid = outp_valid.to(device)

            for _ in tqdm(range(args.train_iterations), desc="Training NN"):
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inp_train, outp_train),
                                                           batch_size=args.mini_batch,
                                                           shuffle=True)
                for inp, target in tqdm(train_loader, desc="Running minibatches"):
                    nn.backpropagation_learn(inp, target, args.learning_rate, show_progress=True, stochastic=False)

            train_err = nn.evaluate(inp_train, outp_train)
            valid_err = nn.evaluate(inp_valid, outp_valid)

            f.write("{} {} {} {}\n".format(optim_err_train, optim_err, train_err, valid_err))
            f.flush()


if __name__ == "__main__":
    main()
