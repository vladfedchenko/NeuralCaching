"""
The script takes a trace as an input and produces a dataset for NN consumption. The dataset consists of rows with the
next information - ID of the object (optionally), popularity during previous and current time windows (calculated using
count-min sketches), time from the start of current time window (optionally), popularity in next time window.
"""
import argparse
import pandas as pd
from tqdm import tqdm
from helpers.collections import CountMinSketch


dataframe_cols = None


def write_row(out_file, row: list):
    """
    Write a row to output file
    :param out_file: Output file.
    :param row: Row to write.
    """
    row = [str(i) for i in row]
    str_to_write = ','.join(row) + '\n'
    out_file.write(str_to_write)


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
    parser.add_argument("-ps",
                        "--prediction_sketches",
                        type=int,
                        help="number of sketches to predict with",
                        default=2)
    parser.add_argument("-sid",
                        "--save_id",
                        help="save id of the object",
                        action="store_true")
    parser.add_argument("-st",
                        "--save_time",
                        help="save fraction of time window from the start of the window when the object arrives",
                        action="store_true")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="output file name",
                        default="dataset_cm.csv")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])

    time_processed = input_df.from_start.min() + args.window_size
    max_time = input_df.iloc[:, 0].max()

    cm_sketches = []
    prev_win_df = None
    with open(args.output, 'w') as out_file:
        with tqdm(total=max_time - time_processed, desc="Processing trace") as prog_bar:
            while time_processed < max_time - args.window_size:
                input_df = input_df[input_df.from_start >= time_processed - args.window_size]
                window_df = input_df[input_df.from_start < time_processed]

                cm_cur = CountMinSketch.construct_by_constraints(0.001, 0.99)

                for index, row in tqdm(window_df.iterrows(), desc="Counting window entries"):
                    id_ = int(row.id)
                    cm_cur.update_counters(id_)

                cm_sketches.append(cm_cur)
                if len(cm_sketches) > args.prediction_sketches + 1:
                    cm_sketches.pop(0)

                if len(cm_sketches) == args.prediction_sketches + 1:
                    prev_win_start = time_processed - 2.0 * args.window_size
                    prev_cm = CountMinSketch.construct_by_constraints(0.001, 0.99)
                    for index, row in tqdm(prev_win_df.iterrows(), desc="Producing DS entries"):
                        to_write = []
                        id_ = int(row.id)
                        prev_cm.update_counters(id_)

                        if args.save_id:
                            to_write.append(id_)

                        for old_cm in cm_sketches[:-2]:
                            to_write.append(old_cm.get_request_fraction(id_))

                        to_write.append(prev_cm.get_request_fraction(id_))

                        if args.save_time:
                            to_write.append((row.from_start - prev_win_start) / float(args.window_size))

                        to_write.append(cm_sketches[-1].get_request_fraction(id_))

                        write_row(out_file, to_write)
                    out_file.flush()

                prev_win_df = window_df
                time_processed += args.window_size
                prog_bar.update(args.window_size)


if __name__ == "__main__":
    main()
