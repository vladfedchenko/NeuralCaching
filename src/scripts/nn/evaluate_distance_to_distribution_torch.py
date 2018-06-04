"""
This script is created to evaluate the accuracy of feedforward neural network by comparing distance of predicted
distribution of popularities to theoretical distribution and by creating artificial cache hit metric.
It is not a multi-use script, it is just a base for multiple cases of distribution.
"""
import pandas as pd
import numpy as np
from neural_nets import TorchFeedforwardNN
import argparse
from tqdm import tqdm
import pickle
from data.generation import PoissonZipfGenerator, PoissonShuffleZipfGenerator
import torch
import torch.utils.data
import os


def calc_aver_error(inp, outp, has_labels):
    tmp = inp
    if has_labels:
        tmp = tmp[:, 1:]

    tmp = np.exp(-tmp) - 10 ** -15  # transform from log
    mean_vals = np.mean(tmp, axis=1)
    mean_vals = -np.log(mean_vals + 10 ** -15)  # transform to log
    # print(tmp[0, :], outp_valid[0, :])

    err = mean_vals - outp
    optim_err = np.mean(np.multiply(err, err))
    return optim_err


def calc_case_2_optim_err(data, has_labels):
    pop1 = data[data.ix[:, 0] <= 5000]
    pop2 = data[data.ix[:, 0] > 5000]

    inp_pop1 = np.matrix(pop1.iloc[:, 1:pop1.shape[1] - 1])
    outp_pop1 = np.matrix(pop1.iloc[:, pop1.shape[1] - 1:pop1.shape[1]])

    outp_pop2 = np.matrix(pop2.iloc[:, pop2.shape[1] - 1:pop2.shape[1]])

    err_sum_pop1 = calc_aver_error(inp_pop1, outp_pop1, has_labels) * len(pop1)

    optim_pop2 = -np.log(10 ** -4 + 10 ** -15)  # transform to log
    err_pop2 = outp_pop2 - optim_pop2
    err_sum_pop2 = np.sum(np.multiply(err_pop2, err_pop2))

    mean_err = (err_sum_pop1 + err_sum_pop2) / len(data)
    return mean_err


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
    parser.add_argument("--case",
                        type=int,
                        help="case of data popularity distribution",
                        default=1)
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
    parser.add_argument("-fc",
                        "--force_cpu",
                        help="force cpu execution for PyTorch",
                        action="store_true")
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
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    data = pd.read_csv(args.input, header=None)

    if args.sample:
        data = data.sample(n=args.sample)

    n = len(data)
    train_size = n * args.train_validation_split

    train_data = data.sample(n=int(train_size))
    valid_data = data.drop(train_data.index)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    print("Running on: {0}".format(device))

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

    sample_map = {}
    for k, v in tqdm(dist_mapping.items(), desc="Preprocessing dataset"):
        sample_map[k] = data[data.ix[:, 0] == k]

    learning_rate = args.learning_rate
    prev_dist = 10**10

    inp_train = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
    outp_train = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])

    inp_valid = np.matrix(valid_data.iloc[:, 1:valid_data.shape[1] - 1])
    outp_valid = np.matrix(valid_data.iloc[:, valid_data.shape[1] - 1:valid_data.shape[1]])

    if args.case == 1:
        optim_err = calc_aver_error(inp_valid, outp_valid, args.input_has_labels)
        optim_err_train = calc_aver_error(inp_train, outp_train, args.input_has_labels)
    elif args.case == 2:
        optim_err = calc_case_2_optim_err(valid_data, args.input_has_labels)
        optim_err_train = calc_case_2_optim_err(train_data, args.input_has_labels)
    else:
        raise AttributeError("Unknown case passed")

    inp_train = torch.from_numpy(inp_train)
    outp_train = torch.from_numpy(outp_train)
    inp_valid = torch.from_numpy(inp_valid)
    outp_valid = torch.from_numpy(outp_valid)

    if torch.cuda.is_available():
        inp_train = inp_train.to(device)
        outp_train = outp_train.to(device)
        inp_valid = inp_valid.to(device)
        outp_valid = outp_valid.to(device)

    dist_file = os.path.join(args.directory, "distance.txt")
    error_file = os.path.join(args.directory, "error.txt")
    with open(error_file, "w") as err_f:
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
            err_f.write("{} {}\n".format(optim_err_train, optim_err))
            for _ in tqdm(range(args.iterations), desc="Running iterations"):
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inp_train, outp_train),
                                                           batch_size=args.mini_batch,
                                                           shuffle=True)
                for inp, target in tqdm(train_loader, desc="Running minibatches"):
                    nn.backpropagation_learn(inp, target, args.learning_rate, show_progress=True, stochastic=False)

                dist = 0.0
                err = 0.0
                for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
                    item = sample_map[k].sample(n=1)
                    inp = torch.from_numpy(np.matrix(item.iloc[:, 1:item.shape[1] - 1]))
                    outp = torch.from_numpy(np.matrix(item.iloc[:, item.shape[1] - 1:item.shape[1]]))

                    err += nn.evaluate(inp, outp)

                    pop = float(nn(torch.Tensor(np.matrix(item.iloc[:, 1:item.shape[1] - 1])).double()))
                    pop = np.exp(-pop) - 10 ** -15

                    dist += abs(v - pop)

                err /= len(dist_mapping)

                dist /= 2.0
                prev_dist = dist

                f.write(f"{dist} {err}\n")
                f.flush()

                train_err = nn.evaluate(inp_train, outp_train)
                valid_err = nn.evaluate(inp_valid, outp_valid)

                err_f.write("{} {}\n".format(train_err, valid_err))
                err_f.flush()

    if args.pickle_file is not None:
        filename = "nn_{0}.p".format(args.pickle_file)
        filename = os.path.join(args.directory, filename)
        with open(filename, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)

    cache_file = os.path.join(args.directory, "cache_hit.txt")
    with open(cache_file, "w") as f:
        popularities = []
        for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
            item = sample_map[k].sample(n=1)
            pop = float(nn(torch.Tensor(np.matrix(item.iloc[:, 1:item.shape[1] - 1])).double()))
            pop = np.exp(-pop) - 10 ** -15

            # tmp = np.matrix(item.iloc[:, 1:item.shape[1] - 1])
            # tmp = np.exp(-tmp) - 10 ** -15  # transform from log
            # pop = float(np.mean(tmp, axis=1))

            # tmp = np.exp(-np.matrix(item.iloc[:, -1:])) - 10 ** -15  # transform from log
            # pop = float(tmp)
            popularities.append((k, pop))

        mean_val = np.mean([x[1] for x in popularities])
        median_val = np.median([x[1] for x in popularities])

        print("Popularity mean: {}".format(mean_val))
        print("Popularity median: {}".format(median_val))

        stat_file = os.path.join(args.directory, "stat.txt")
        with open(stat_file, "w") as f_stat:
            f_stat.write("Popularity mean: {}".format(mean_val))
            f_stat.write("Popularity median: {}".format(median_val))

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


if __name__ == "__main__":
    main()
