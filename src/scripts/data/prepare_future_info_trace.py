"""
This script takes a trace as an input and produces a new trace in which each row contains the popularity of the object
in the next time window.
"""
import argparse
import pandas as pd
from tqdm import tqdm
from helpers.collections import FullCounter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input trace file")
    parser.add_argument("-w",
                        "--window_size",
                        type=int,
                        help="time window size",
                        default=300_000)
    parser.add_argument("-hwp",
                        "--half_window_prediction",
                        help="make the output popularity at the end of current window until middle of window reached",
                        action="store_true")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="output file name",
                        default="dataset_cm.csv")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])

    time_processed = input_df.from_start.min()
    max_time = input_df.iloc[:, 0].max()

    cur_win_counter = None

    prev_win_df = None
    with open(args.output, 'w') as out_file:
        with tqdm(total=max_time - time_processed, desc="Processing trace") as prog_bar:
            while time_processed < max_time + args.window_size:
                input_df = input_df[input_df.from_start >= time_processed]
                window_df = input_df[input_df.from_start < time_processed + args.window_size]

                if len(window_df) > 0:
                    next_win_counter = FullCounter()
                    for index, row in tqdm(window_df.iterrows(), desc="Counting window entries"):
                        id_ = int(row.id)
                        next_win_counter.update_counters(id_)
                else:
                    next_win_counter = cur_win_counter

                if cur_win_counter is None:
                    cur_win_counter = next_win_counter
                    time_processed += args.window_size
                    prev_win_df = window_df
                    continue

                win_start_time = None
                for index, row in tqdm(prev_win_df.iterrows(), desc="Producing new entries"):
                    if win_start_time is None:
                        win_start_time = row.from_start
                    t = (row.from_start - win_start_time) / args.window_size
                    # print("\n\nt: {}\nfrom_start: {}\ntime_processed: {}".format(t, row.from_start, time_processed))
                    assert 0.0 <= t <= 1.0

                    if args.half_window_prediction and t <= 0.5:
                        next_pop = cur_win_counter.get_request_fraction(int(row.id))
                    else:
                        next_pop = next_win_counter.get_request_fraction(int(row.id))

                    out_file.write("{}, {}, {}, {}\n".format(row.from_start, row.from_prev, int(row.id), next_pop))
                out_file.flush()

                time_processed += args.window_size
                prev_win_df = window_df
                cur_win_counter = next_win_counter
                prog_bar.update(args.window_size)


if __name__ == "__main__":
    main()
