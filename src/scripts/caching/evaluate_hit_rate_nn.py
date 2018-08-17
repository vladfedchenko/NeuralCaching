"""
This script is intended to evaluate cache hit rate throughout different cache sizes.
"""
import argparse
from tqdm import tqdm
import pickle
from caching import *
import json
import torch
from typing import List
import math


def construct_metadata(row: List[str], args: argparse.Namespace) -> dict:
    ret = {}
    if args.size_meta:
        ret["size"] = int(row[1])

    if args.daytime_meta:
        ret["daytime"] = (float(row[0]) % 86400.0) / 86400.0 * 2 * math.pi

    return ret


def eval_cache_hit(cache: AbstractCache, trace_file: str, cold_start_skip: int, log_file: str = None) -> float:
    """
    Evaluate cache hit on some trace.
    :param cache: Cache object.
    :param trace_file: File with trace.
    :param cold_start_skip: Skip a number of items when evaluating hit rate to avoid cold start.
    :param log_file: Log file name to write intermediate results.
    :return: Cache hit rate.
    """
    requests = 0
    hits = 0.0

    instant_hits = 0.0
    instant_requests = 0

    log = None
    if log_file is not None:
        log = open(log_file, 'w')

    with open(trace_file, 'r') as trace:
        for i, row in tqdm(enumerate(trace), desc="Running trace"):
            row = row.split(',')
            if i < cold_start_skip:
                cache.request_object(int(row[2]), 1, float(row[0]), {"size": int(row[1])})
                continue

            requests += 1
            instant_requests += 1
            if cache.request_object(int(row[2]), 1, float(row[0]), {"size": int(row[1])}):
                hits += 1.0
                instant_hits += 1.0

            if log is not None and requests > 100 and i % 10**6 == 0:
                log.write("{} {} {} {}\n".format(len(cache), i, hits / requests, instant_hits / instant_requests))
                log.flush()

                instant_hits = 0.0
                instant_requests = 1

    if log is not None:
        log.close()

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
    parser.add_argument("cache_descriptor",
                        type=str,
                        help="descriptor to construct cache object")
    parser.add_argument("output",
                        help="output file name",
                        type=str)
    parser.add_argument("-log",
                        "--log_file_prefix",
                        help="log files prefix to write intermediate results",
                        type=str,
                        default=None)
    parser.add_argument("-css",
                        "--cold_start_skip",
                        help="number of requests to skip when evaluating hit rate",
                        type=int,
                        default=0)
    parser.add_argument("-fc",
                        "--force_cpu",
                        help="force cpu execution for PyTorch",
                        action="store_true")
    parser.add_argument("-sm",
                        "--size_meta",
                        help="add size to metadata for NN",
                        action="store_true")
    parser.add_argument("-dm",
                        "--daytime_meta",
                        help="add daytime to metadata for NN",
                        action="store_true")
    args = parser.parse_args()

    with tqdm(total=args.max_cache, desc="Sizes processed") as pbar:
        cur_size = args.starting_cache
        with open(args.output, 'w') as f:
            while cur_size <= args.max_cache:

                desc = None
                if args.cache_descriptor is not None:
                    with open(args.cache_descriptor) as f_desc:
                        desc = json.load(f_desc)

                nn = None
                if desc["nn_location"] != "":
                    with open(desc["nn_location"], "rb") as unpickle_file:
                        nn = pickle.load(unpickle_file)

                if not args.force_cpu and torch.cuda.is_available():
                    print("Running on: GPU")
                    torch.set_default_tensor_type("torch.cuda.FloatTensor")
                else:
                    print("Running on: CPU")
                    torch.set_default_tensor_type("torch.FloatTensor")

                online = (desc["online_learning"] == "True")
                cache = FeedforwardNNCacheFullTorch(cur_size,
                                                    nn,
                                                    int(desc["counter_num"]),
                                                    float(desc["time_window"]),
                                                    int(desc["update_sample_size"]),
                                                    online,
                                                    float(desc["cf_coef"]),
                                                    float(desc["learning_rate"]),
                                                    int(desc["batch_size"]))

                hit_rate = eval_cache_hit(cache,
                                          args.input,
                                          args.cold_start_skip,
                                          args.log_file + "_{}.log".format(cur_size))

                f.write(f"{cur_size} {hit_rate}\n")
                f.flush()

                cur_size += args.size_increment
                pbar.update(args.size_increment)


if __name__ == "__main__":
    main()
