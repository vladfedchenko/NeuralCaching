"""
This script is created to prepare the dataset for the neural network consumption.
The basic idea is to provide the synthetic trace, split it into time windows, calculate the popularity of each item
through each time window. Then create a dataset in which rows are K popularity values through K time windows.
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


def fill_res_dataframe(out_file, ids: list, input_cms_list: list, output_cms: CountMinSketch):
    """
    Fill the result dataframe with calculated with count-min sketches and item popularities.
    :param out_file: Output file name
    :param ids: List of IDs of all the items.
    :param input_cms_list: list(CountMinSketch) - calculated input count-min sketches.
    :param output_cms: Output count-min sketch
    """
    global dataframe_cols
    for id_ in tqdm(ids):
        row = [id_]

        counts = []
        count_min_states = []
        for cms in input_cms_list:
            counts.append(cms.get_count(id_))
            count_min_states.extend(cms.get_counter_state().tolist()[0])

        counts.extend(count_min_states)

        row.append(len(counts))
        row.extend(counts)

        row.append(output_cms.get_count(id_))
        row.extend(list(output_cms.get_counter_state().tolist())[0])

        if dataframe_cols is None:
            cols = [''] * len(row)
            cols[0] = 'id'
            cols[1] = 'input_size'
            dataframe_cols = cols
            write_row(out_file, cols)
        write_row(out_file, row)


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
    parser.add_argument("-g",
                        "--window_group_size",
                        type=int,
                        help="result dataset column number",
                        default=2)
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="output file name. \"dataset_mixed.csv\" otherwise",
                        default="dataset_mixed.csv")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, header=None, names=["from_start", "from_prev", "id"])

    ids = list(sorted(input_df.id.unique()))
    time_processed = 0
    max_time = input_df.from_start.max()

    cm_sketches = []
    cm_goal_count = args.window_group_size * 2 + 2  # + 2 is to have the goal output
    half_window = args.window_size / 2
    with open(args.output, 'w') as out_file:
        with tqdm(total=max_time) as prog_bar:
            while time_processed < max_time:
                input_df = input_df[input_df.from_start >= time_processed]
                window_df = input_df[input_df.from_start < time_processed + half_window]

                cm_cur = CountMinSketch.construct_by_constraints(0.01)
                cm_prev = None
                if len(cm_sketches) > 0:
                    cm_prev = cm_sketches[-1]

                for index, row in window_df.iterrows():
                    id_ = int(row.id)
                    cm_cur.update_counters(id_)
                    if cm_prev is not None:
                        cm_prev.update_counters(id_)

                cm_sketches.append(cm_cur)
                if len(cm_sketches) > cm_goal_count:
                    cm_sketches.pop(0)

                if len(cm_sketches) == cm_goal_count:
                    fill_res_dataframe(out_file, ids, cm_sketches[0:args.window_group_size + 1], cm_sketches[-2])

                time_processed += half_window
                prog_bar.update(half_window)


if __name__ == "__main__":
    main()
