"""
This script is intended to evaluate the error produced by introduction of Count-Min sketch when calculating item
popularity.
"""
import argparse
import pandas as pd
import os
from helpers.collections import CountMinSketch
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input trace file")
    parser.add_argument("directory",
                        type=str,
                        help="directory to store data files")
    parser.add_argument("-w",
                        "--window_size",
                        type=int,
                        help="time window size",
                        default=1_000_000)
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])
    # input_df = pd.read_csv(args.input, header=None, names=["from_start", "size", "id"])

    ids = list(sorted(input_df.id.unique()))
    max_time = input_df.from_start.max()

    cm_size_range = list(range(10, 91, 10)) + list(range(100, 901, 100)) + list(range(1000, 10001, 1000))

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    out = os.path.join(args.directory, "cm_error.txt")
    with open(out, 'w') as f:
        for cm_size in tqdm(cm_size_range, desc="Processing CM sizes"):
            err_list = []
            time_processed = input_df.from_start.min() + args.window_size
            with tqdm(total=max_time - time_processed, desc="Time processed", unit="unit(s)") as pbar:

                while time_processed < max_time:
                    window_df = input_df[(input_df.from_start >= (time_processed - args.window_size)) &
                                         (input_df.from_start < time_processed)]
                    cm = CountMinSketch.construct_by_space(cm_size)
                    pop_map = {}
                    for id_ in tqdm(ids, desc="Calculating item popularity"):
                        id_int = int(id_)
                        id_df = window_df[window_df.id == id_int]
                        pop_map[id_int] = len(id_df)
                        cm.update_counters(id_int, pop_map[id_int])

                    for id_ in tqdm(ids, desc="Calculating error"):
                        id_int = int(id_)
                        err = float(cm.get_count(id_int) - pop_map[id_int])
                        assert err >= 0.0
                        err_list.append(err)

                    time_processed += args.window_size
                    pbar.update(args.window_size)

            mean_abs_err = np.mean(np.abs(err_list))
            mean_sqr_err = np.mean(np.multiply(err_list, err_list))

            f.write("{} {} {}\n".format(cm_size, mean_abs_err, mean_sqr_err))
            f.flush()


if __name__ == "__main__":
    main()
