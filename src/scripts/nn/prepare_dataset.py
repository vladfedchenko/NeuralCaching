"""
This script is created to prepare the dataset for the neural network consumption.
The basic idea is to provide the synthetic trace, split it into time windows, calculate the popularity of each item
through each time window. Then create a dataset in which rows are K popularity values through K time windows.
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input trace file")
    parser.add_argument("-w",
                        "--window_size",
                        type=int,
                        help="time window size",
                        default=50_000)
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
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])

    ids = list(sorted(input_df.id.unique()))
    time_processed = 0
    max_time = input_df.from_start.max()
    frame = 1
    full_popularity_df = pd.DataFrame(index=ids)
    while time_processed < max_time:
        input_df = input_df[input_df.from_start >= time_processed]
        window_df = input_df[input_df.from_start < time_processed + args.window_size]
        items_count = float(len(window_df))
        pop_col = []
        for id_ in ids:
            id_df = window_df[window_df.id == id_]
            pop_col.append(len(id_df) / items_count)

        full_popularity_df[str(frame)] = pop_col
        frame += 1
        time_processed += args.window_size

    res_df = pd.DataFrame()
    ful_pop_cols = full_popularity_df.shape[1]
    cur_col = args.window_group_size
    while cur_col <= ful_pop_cols:
        to_add = full_popularity_df.iloc[:, cur_col - args.window_group_size:cur_col]
        to_add.columns = range(1, args.window_group_size + 1)
        res_df = pd.concat([res_df, to_add], ignore_index=True)
        cur_col += 1

    res_df.to_csv(args.output)


if __name__ == "__main__":
    main()
