"""
This script is intended to evaluate cache hit rate throughout different cache sizes.
"""
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
from caching import *
from caching.abstract_cache import AbstractCache
import json


def eval_cache_hit(cache: AbstractCache, trace: pd.DataFrame, cold_start_skip: int) -> float:
    """
    Evaluate cache hit on some trace.
    :param cache: Cache object.
    :param trace: Object trace.
    :return: Cache hit rate
    """
    requests = 0
    hits = 0.0
    i = 0
    for _, row in tqdm(trace.iterrows(), desc="Running trace", total=len(trace)):

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
    parser.add_argument("cache_type",
                        type=str,
                        help="type of caching policy to use")
    parser.add_argument("output",
                        help="output file name",
                        type=str)
    parser.add_argument("-cd",
                        "--cache_descriptor",
                        type=str,
                        help="descriptor to construct cache object")
    parser.add_argument("-css",
                        "--cold_start_skip",
                        help="number of requests to skip when evaluating hit rate",
                        type=int,
                        default=0)
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])
    # input_df = input_df.iloc[:1_000_000, :]
    # print(input_df.shape)

    with tqdm(total=args.max_cache, desc="Sizes processed") as pbar:
        cur_size = args.starting_cache
        with open(args.output, 'w') as f:
            while cur_size <= args.max_cache:

                desc = None
                if args.c is not None:
                    with open(args.cache_descriptor) as f:
                        desc = json.load(f)

                if args.cache_type == "nn":
                    with open(desc["nn_location"], "rb") as unpickle_file:
                        nn = pickle.load(unpickle_file)
                    cache = FeedforwardNNCacheFullTorch(cur_size,
                                                        nn,
                                                        int(desc["counter_num"]),
                                                        float(desc["time_window"]),
                                                        int(desc["update_sample_size"]))

                elif args.cache_type == "future":
                    cache = FutureInfoCache(cur_size, desc["mod_trace_path"])

                elif args.cache_type == "lru":
                    cache = LRUCache(cur_size)

                elif args.cache_type == "arc":
                    cache = ARCache(cur_size)

                else:
                    raise Exception("Unidentified error type")

                hit_rate = eval_cache_hit(cache, input_df, args.cold_start_skip)
                f.write(f"{cur_size} {hit_rate}\n")
                f.flush()

                cur_size += args.size_increment
                pbar.update(args.size_increment)


if __name__ == "__main__":
    main()
