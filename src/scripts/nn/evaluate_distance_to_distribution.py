"""
This script is created to evaluate the accuracy of feedforward neural network by comparing distance of predicted
distribution of popularities to theoretical distribution and by creating artificial cache hit metric.
It is not a multi-use script, it is just a base for multiple cases of distribution.
"""
import pandas as pd
import numpy as np
from neural_nets.feedforward_nn import FeedforwardNeuralNet, sigmoid, sigmoid_deriv
import argparse
from tqdm import tqdm
import pickle
from data.generation import PoissonZipfGenerator


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
                        help="number of samples to use on learning step. If not passed - whole dataset is used",
                        default=None)
    parser.add_argument("-od",
                        "--output_file_distance",
                        type=str,
                        help="file to output distance values",
                        default="distance.txt")
    parser.add_argument("-oc",
                        "--output_cache_file",
                        type=str,
                        help="file to output cache hit values",
                        default="cache hit.txt")
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
    parser.add_argument("-mln",
                        "--middle_layer_neurons",
                        type=int,
                        help="middle layer neuron count",
                        default=20)
    args = parser.parse_args()

    # In the next section you should define a mapping of items distribution

    generator = PoissonZipfGenerator(100_000, 20.0, 0.8, 0)
    dist_mapping = generator.get_distribution_map()

    # End of section

    data = pd.read_csv(args.input, header=None)

    if args.unpickle_file is not None:
        with open(args.unpickle_file, "rb") as unpickle_file:
            nn = pickle.load(unpickle_file)
    else:
        nn = FeedforwardNeuralNet([data.shape[1] - 2, args.middle_layer_neurons, 1],
                                  out_activ=sigmoid,
                                  out_activ_deriv=sigmoid_deriv)

    sample_map = {}
    for k, v in tqdm(dist_mapping.items(), desc="Preprocessing dataset"):
        sample_map[k] = data[data.ix[:, 0] == k]

    with open(args.output_file_distance, "w") as f:
        for _ in tqdm(range(args.iterations), desc="Running iterations"):
            if args.train_sample_size is None:
                train_data = data
            else:
                train_data = data.sample(n=args.train_sample_size)
            inp = np.matrix(train_data.iloc[:, 1:train_data.shape[1] - 1])
            outp = np.matrix(train_data.iloc[:, train_data.shape[1] - 1:train_data.shape[1]])
            nn.backpropagation_learn(inp, outp, args.learning_rate, show_progress=True)

            dist = 0.0
            for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
                item = sample_map[k].sample(n=1)
                pop = nn.evaluate(np.matrix(item.iloc[:, 1:item.shape[1] - 1]),
                                  np.matrix(item.iloc[:, item.shape[1] - 1:item.shape[1]]))[0]

                dist += abs(v - pop)

            dist /= 2.0

            f.write(f"{dist}\n")

    with open(args.output_file_distance, "w") as f:
        popularities = []
        for k, v in tqdm(dist_mapping.items(), desc="Evaluating distance"):
            item = data[data.ix[:, 0] == k].sample(n=1)
            pop = nn.evaluate(np.matrix(item.iloc[:, 1:item.shape[1] - 1]),
                              np.matrix(item.iloc[:, item.shape[1] - 1:item.shape[1]]))[0]
            popularities.append((k, pop))

        pops_sorted = list(sorted(popularities, key=lambda x: x[1], reverse=True))
        pop_order_predicted = [x[0] for x in pops_sorted]
        pred_items_real_pops = [dist_mapping[i] for i in pop_order_predicted]

        distrib_pop_ordered = sorted(dist_mapping.values(), reverse=True)

        theory_hit = 0.0
        practice_hit = 0.0
        for distrib_pop, pred_item_pop in zip(distrib_pop_ordered, pred_items_real_pops):
            theory_hit += distrib_pop
            practice_hit += pred_item_pop
            f.write(f"{theory_hit} {practice_hit}\n")

    if args.pickle_file is not None:
        with open(args.pickle_file, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)


if __name__ == "__main__":
    main()
