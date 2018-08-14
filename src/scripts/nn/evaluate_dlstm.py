import argparse
import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from neural_nets import LSTMSoftmax
import pickle
from tqdm import tqdm


inputs_num = 5
outputs_num = 100


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
    parser.add_argument("-mbl",
                        "--mini_batch_log",
                        type=int,
                        help="after how many batches evaluate the error",
                        default=100)
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
    parser.add_argument("--seed",
                        help="seed for item sampling",
                        type=int)
    parser.add_argument("-fc",
                        "--force_cpu",
                        help="force cpu execution for PyTorch",
                        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    data = pd.read_csv(args.input, header=None, index_col=None, names=None)

    if args.sample:
        data = data.sample(n=args.sample)

    n = len(data)
    train_size = n * args.train_validation_split

    train_data = data.sample(n=int(train_size))
    valid_data = data.drop(train_data.index)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    print("Running on: {0}".format(device))

    if args.unpickle_file is not None:
        filename = "dlstm_{0}.p".format(args.unpickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        layers = [inputs_num, 16, 16, outputs_num]
        nn = LSTMSoftmax(layers)
        if torch.cuda.is_available():
            nn.to(device)

    inp_train = np.matrix(train_data.iloc[:, :inputs_num]).astype(float)
    outp_train = np.matrix(train_data.iloc[:, inputs_num:])

    inp_valid = np.matrix(valid_data.iloc[:, :inputs_num]).astype(float)
    outp_valid = np.matrix(valid_data.iloc[:, inputs_num:])

    #torch.set_default_dtype(torch.float64)

    inp_train = torch.from_numpy(inp_train).type(torch.FloatTensor)
    outp_train = torch.from_numpy(outp_train).type(torch.FloatTensor)
    inp_valid = torch.from_numpy(inp_valid).type(torch.FloatTensor)
    outp_valid = torch.from_numpy(outp_valid).type(torch.FloatTensor)

    if torch.cuda.is_available():
        inp_train = inp_train.to(device)
        outp_train = outp_train.to(device)
        inp_valid = inp_valid.to(device)
        outp_valid = outp_valid.to(device)

    log_counter = args.mini_batch_log
    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "w") as f:
        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inp_train, outp_train),
                                                       batch_size=args.mini_batch,
                                                       shuffle=True)
            for inp, target in tqdm(train_loader, desc="Running minibatches"):
                nn.backpropagation_learn(inp, target, args.learning_rate, show_progress=True, stochastic=False)
                log_counter -= 1

                if log_counter == 0:
                    log_counter = args.mini_batch_log
                    train_err = nn.evaluate(inp_train, outp_train)
                    valid_err = nn.evaluate(inp_valid, outp_valid)

                    f.write("{} {}\n".format(train_err, valid_err))
                    f.flush()

    if args.pickle_file is not None:
        filename = "dlstm_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
