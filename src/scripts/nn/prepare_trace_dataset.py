"""
This script is created to prepare the dataset for the neural network consumption.
The basic idea is to provide the synthetic trace, split it into time windows, calculate the popularity of each item
through each time window. Then create a dataset in which rows are K popularity values through K time windows.
"""
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np


def write_batch(out_file, data_matrix):
    for line in data_matrix:
        str_line = [str(i) for i in line.flat]
        str_to_write = ','.join(str_line) + '\n'
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
                        default=1_000_000)
    parser.add_argument("-g",
                        "--window_group_size",
                        type=int,
                        help="result dataset column number",
                        default=5)
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="output file name. \"dataset.csv\" otherwise",
                        default="dataset.csv")
    parser.add_argument("-cs",
                        "--class_separator",
                        help=("class separation value, basically number of items in first population if population is " 
                              "mixed. If passed then every row will have class label"),
                        type=int,
                        default=None)
    parser.add_argument("-sid",
                        "--save_id",
                        help="save id of the object")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])

    ids = list(sorted(input_df.id.unique()))
    max_time = input_df.from_start.max()

    time_processed = input_df.from_start.min()

    ids_class = None
    if args.class_separator is not None:
        ids_class = list(map(lambda x: int(x > args.class_separator), ids))

    window_list = []
    with tqdm(total=max_time - time_processed, desc="Time processed", unit="unit(s)") as pbar:
        with open(args.output, 'w') as f:
            while time_processed < max_time:
                input_df = input_df[input_df.from_start >= time_processed]
                window_df = input_df[input_df.from_start < time_processed + args.window_size]
                items_count = float(len(window_df))
                pop_col = []
                for id_ in ids:
                    id_df = window_df[window_df.id == id_]
                    if items_count != 0.0:
                        pop_col.append(len(id_df) / items_count)
                    else:
                        pop_col.append(0.0)

                window_list.append(pop_col)

                if len(window_list) > args.window_group_size:
                    del window_list[0]

                if len(window_list) == args.window_group_size:
                    data_matrix = np.matrix(window_list).T
                    if ids_class is not None:
                        ids_class_matrix = np.matrix([ids_class]).T
                        data_matrix = np.concatenate((ids_class_matrix, data_matrix), axis=1)

                    if args.save_id:
                        ids_matr = np.matrix([ids]).T
                        data_matrix = np.concatenate((ids_matr, data_matrix), axis=1)

                    write_batch(f, data_matrix)

                time_processed += args.window_size
                pbar.update(args.window_size)


if __name__ == "__main__":
    main()
