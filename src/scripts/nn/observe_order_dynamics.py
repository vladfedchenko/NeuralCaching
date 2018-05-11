"""
This script is designed to create multiple "order" files that NN predicts.
"""
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
from data.generation import PoissonZipfGenerator, PoissonShuffleZipfGenerator
from neural_nets.feedforward_nn import FeedforwardNeuralNet, sigmoid, sigmoid_deriv
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input dataset")
    parser.add_argument("directory",
                        type=str,
                        help="directory with data files")
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
    parser.add_argument("-ts",
                        "--train_sample_size",
                        type=int,
                        help="number of samples to use on learning step. If not passed - whole dataset is used",
                        default=None)
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
    parser.add_argument("--case",
                        type=int,
                        help="case of data popularity distribution",
                        default=1)
    parser.add_argument("-shl",
                        "--sigmoid_hidden_layers",
                        help="use sigmoid on hidden layers",
                        action="store_true")
    parser.add_argument("-sol",
                        "--sigmoid_output_layers",
                        help="use sigmoid on output layer",
                        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Case 1
    if args.case == 1:
        generator = PoissonZipfGenerator(10_000, 20.0, 0.8, 0)
        dist_mapping = generator.get_distribution_map()

    # Case 2
    elif args.case == 2:
        generator = PoissonZipfGenerator(5_000, 40.0, 0.8, 0)
        dist_mapping = generator.get_distribution_map()

        generator2 = PoissonShuffleZipfGenerator(5_000, 40.0, 0.8, 5_000, 10_000_000)
        dist_mapping2 = generator2.get_distribution_map()
        for k, v in dist_mapping2.items():
            dist_mapping[k] = v

        for k, v in dist_mapping.items():
            dist_mapping[k] = v / 2.0

    else:
        raise AttributeError("Unknown case passed")

    data = pd.read_csv(args.input, header=None)

    if args.unpickle_file is not None:
        filename = "order_nn_{0}.p".format(args.unpickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        act_hid = None
        act_hid_deriv = None
        act_out = None
        act_out_deriv = None

        if args.sigmoid_hidden_layers:
            act_hid = sigmoid
            act_hid_deriv = sigmoid_deriv

        if args.sigmoid_output_layers:
            act_out = sigmoid
            act_out_deriv = sigmoid_deriv

        layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
        nn = FeedforwardNeuralNet(layers,
                                  internal_activ=act_hid,
                                  internal_activ_deriv=act_hid_deriv,
                                  out_activ=act_out,
                                  out_activ_deriv=act_out_deriv)

    assert(nn is not None)

    sample_map = {}
    for k, v in tqdm(dist_mapping.items(), desc="Preprocessing dataset"):
        sample_map[k] = data[data.ix[:, 0] == k]

    for i in tqdm(range(args.iterations + 1), desc="Running iterations"):

        filename = "{0}{1}.txt".format("order_", i)
        filename = os.path.join(args.directory, filename)

        with open(filename, "w") as f:
            popularities = []
            for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
                item = sample_map[k].sample(n=1)
                pop = float(nn.feedforward(np.matrix(item.iloc[:, 1:item.shape[1] - 1]).T))
                # pop = float(np.mean(np.matrix(item.iloc[:, 1:item.shape[1] - 1])))
                popularities.append((k, pop))

            pops_sorted = list(sorted(popularities, key=lambda x: x[1], reverse=True))

            for item in pops_sorted:
                f.write("{0} {1} {2}\n".format(item[0], item[1], dist_mapping[item[0]]))

        if args.train_sample_size is None:
            train_data = data
        else:
            train_data = data.sample(n=args.train_sample_size)
        inp = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
        outp = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])
        nn.backpropagation_learn(inp, outp, args.learning_rate, show_progress=True, stochastic=True)

    if args.pickle_file is not None:
        filename = "order_nn_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
