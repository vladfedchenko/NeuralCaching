"""
This script is created to evaluate the accuracy of feedforward neural network by comparing distance of predicted
distribution of popularities to theoretical distribution and by creating artificial cache hit metric.
It is not a multi-use script, it is just a base for multiple cases of distribution.
"""
import pandas as pd
import numpy as np
from neural_nets import FeedforwardNeuralNet
import argparse
from tqdm import tqdm
import pickle
from data.generation import PoissonZipfGenerator, PoissonShuffleZipfGenerator
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
    parser.add_argument("-alr",
                        "--adaptive_learning",
                        help="use adaptive learning rate - decrease rate if error went up",
                        action="store_true")
    parser.add_argument("-ha",
                        "--hidden_activation",
                        help="activation to use on hidden layers",
                        type=str)
    parser.add_argument("-oa",
                        "--out_activation",
                        help="activation to use on out layer",
                        type=str)
    parser.add_argument("-ef",
                        "--error_function",
                        help="error function to use",
                        type=str)
    parser.add_argument("--seed",
                        help="seed for item sampling",
                        type=int)
    # parser.add_argument("-aef",
    #                     "--alternative_error_function",
    #                     help="use alternative error function - error for Poisson distribution",
    #                     action="store_true")
    args = parser.parse_args()

    # In the next section you should define a mapping of items distribution

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

    # End of section

    if args.seed:
        np.random.seed(args.seed)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    data = pd.read_csv(args.input, header=None)

    if args.unpickle_file is not None:
        filename = "distance_nn_{0}.p".format(args.unpickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        layers = [data.shape[1] - 2] + ([args.middle_layer_neurons] * args.middle_layers) + [1]
        nn = FeedforwardNeuralNet(layers,
                                  hidden_activation=args.hidden_activation,
                                  out_activation=args.out_activation,
                                  error_func=args.error_function)

    sample_map = {}
    for k, v in tqdm(dist_mapping.items(), desc="Preprocessing dataset"):
        sample_map[k] = data[data.ix[:, 0] == k]

    learning_rate = args.learning_rate
    prev_dist = 10**10

    dist_file = os.path.join(args.directory, "distance.txt")
    with open(dist_file, "w") as f:

        # dist = 0.0
        # for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
        #     item = sample_map[k].sample(n=1)
        #     pop = nn.evaluate(np.matrix(item.iloc[:, 1:item.shape[1] - 1]),
        #                       np.matrix(item.iloc[:, item.shape[1] - 1:item.shape[1]]))[0]
        #
        #     dist += abs(v - pop)
        #
        # dist /= 2.0
        # f.write(f"{dist}\n")
        # f.flush()

        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            if args.train_sample_size is None:
                train_data = data
            else:
                train_data = data.sample(n=args.train_sample_size)
            inp = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
            outp = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])
            nn.backpropagation_learn(inp, outp, args.learning_rate, show_progress=True, stochastic=True)

            dist = 0.0
            err = 0.0
            for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
                item = sample_map[k].sample(n=1)
                inp = np.matrix(item.iloc[:, 1:item.shape[1] - 1])
                outp = np.matrix(item.iloc[:, item.shape[1] - 1:item.shape[1]])

                err += nn.evaluate(inp, outp, show_progress=False)[0]

                pop = float(nn.feedforward(np.matrix(item.iloc[:, 1:item.shape[1] - 1]).T))

                dist += abs(v - pop)

            err /= len(dist_mapping)

            dist /= 2.0
            if args.adaptive_learning and dist > prev_dist:
                learning_rate /= 2.0
            prev_dist = dist

            f.write(f"{dist} {err}\n")
            f.flush()

    cache_file = os.path.join(args.directory, "cache_hit.txt")
    with open(cache_file, "w") as f:
        popularities = []
        for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
            item = sample_map[k].sample(n=1)
            m = np.matrix(item.iloc[:, 1:item.shape[1] - 1]).T
            pop = float(nn.feedforward(m))
            pop = np.exp(pop) - 10 ** -8
            # m = np.exp(m) - 10 ** -8
            # pop = float(np.mean(m))
            popularities.append((k, pop))

        pops_sorted = list(sorted(popularities, key=lambda x: x[1], reverse=True))
        pop_order_predicted = [x[0] for x in pops_sorted]

        order_file = os.path.join(args.directory, "order.txt")
        with open(order_file, "w") as f1:
            for item in pops_sorted:
                f1.write("{0} {1} {2}\n".format(item[0], item[1], dist_mapping[item[0]]))

        pred_items_real_pops = [dist_mapping[i] for i in pop_order_predicted]

        distrib_pop_ordered = sorted(dist_mapping.values(), reverse=True)

        theory_hit = 0.0
        practice_hit = 0.0
        for distrib_pop, pred_item_pop in zip(distrib_pop_ordered, pred_items_real_pops):
            theory_hit += distrib_pop
            practice_hit += pred_item_pop
            f.write(f"{theory_hit} {practice_hit}\n")

    if args.pickle_file is not None:
        filename = "distance_nn_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
