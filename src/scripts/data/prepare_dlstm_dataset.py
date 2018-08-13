"""
This script creates a dataset which is then used by a DLSTM in a way similar to DLSTM-Cache.
"""
import argparse
from tqdm import tqdm
import numpy as np


def item_priority(index, out_len, alpha):
    ret = 1.0 - ((index + 1.0) / out_len) ** alpha
    return ret


def calc_priority(req_sequence, out_len, id_map, alpha):
    priority = [0.0] * out_len
    for i, item in enumerate(req_sequence):
        priority[id_map[item]] += item_priority(i, out_len, alpha)

    priority = np.exp(priority)
    priority /= np.sum(priority, axis=0)

    return priority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="input trace file")
    parser.add_argument("output",
                        type=str,
                        help="output file name")
    parser.add_argument("input_size",
                        type=int,
                        help="size of the input")
    parser.add_argument("output_size",
                        type=int,
                        help="size of the output")
    parser.add_argument("true_pred_seq_len",
                        type=int,
                        help="sequence length to calculate true caching priority")
    parser.add_argument("-a",
                        "--alpha",
                        type=float,
                        help="alpha for determining real caching priority",
                        default=0.125)
    args = parser.parse_args()

    unique_ids = args.output_size
    id_map = {}

    request_log = []

    with open(args.input, 'r') as trace:
        with open(args.output, "w") as out_file:
            for i, row in tqdm(enumerate(trace), desc="Running trace"):
                row = row.split(',')
                id_ = int(row[2])

                if id_ not in id_map:
                    unique_ids -= 1
                    id_map[id_] = unique_ids

                    if unique_ids == -1:
                        raise Exception("Too many unique objects to handle. Consider increasing the output size.")

                request_log.append(id_)
                if len(request_log) < args.input_size + args.true_pred_seq_len:
                    continue

                if len(request_log) > args.input_size + args.true_pred_seq_len:
                    request_log = request_log[1:]

                inp = request_log[:args.input_size]
                priority = calc_priority(request_log[args.input_size:], args.output_size, id_map, args.alpha)

                inp += priority

                inp = [str(i) for i in inp]
                str_to_write = ','.join(inp) + '\n'
                out_file.write(str_to_write)


if __name__ == "__main__":
    main()
