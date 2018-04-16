"""
This script is intended to evaluate cache hit rate throughout different cache sizes.
"""
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
from caching import FeedforwardNNCache, LFUCache, LRUCache, ARCache
from caching.abstract_cache import AbstractCache


def eval_cache_hit(cache: AbstractCache, trace: pd.DataFrame) -> float:
    """
    Evaluate cache hit on some trace.
    :param cache: Cache object.
    :param trace: Object trace.
    :return: Cache hit rate
    """
    requests = 0
    hits = 0.0
    for i, row in tqdm(trace.iterrows(), desc="Running trace"):
        requests += 1
        if cache.request_object(row.id, 1, row.from_start):
            hits += 1.0
    return hits / requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input trace")
    parser.add_argument("starting_cache",
                        type=int,
                        help="starting cache size")
    parser.add_argument("size_increment",
                        type=int,
                        help="increment in cache size")
    parser.add_argument("max_cache",
                        type=int,
                        help="max cache size")
    parser.add_argument("output",
                        help="output file name",
                        type=str)
    args = parser.parse_args()

    with open("case1_count_min_nn.p", "rb") as unpickle_file:
        nn = pickle.load(unpickle_file)

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])
    input_df = input_df.iloc[:1_000_000, :]
    print(input_df.shape)

    with tqdm(total=args.max_cache, desc="Sizes processed") as pbar:
        pbar.update(args.starting_cache)
        cur_size = args.starting_cache
        with open(args.output, 'w') as f:
            while cur_size <= args.max_cache:
                # Feel free to change the type of cache to evaluate

                cache = FeedforwardNNCache(cur_size, nn, 4, 0.001, 0.99, 1_000_000.0, 5)

                # cache = LRUCache(cur_size)

                hit_rate = eval_cache_hit(cache, input_df)
                f.write(f"{hit_rate}\n")

                cur_size += args.size_increment
                pbar.update(args.size_increment)


if __name__ == "__main__":
    main()
