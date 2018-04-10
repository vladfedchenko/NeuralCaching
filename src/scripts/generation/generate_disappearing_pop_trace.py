"""
This script generates a synthetic trace of population with Poisson arrivals and Zipf popularity,
but items can randomly disappear and reappear according to some Poisson process.
The file is generated in CSV format.
"""
import argparse
from tqdm import tqdm
from data.generation import DisappearingPoissonZipfGenerator
import numpy as np

from_prev_list = []


def write_batch(f, batch):
    global from_prev_list
    for tfs, tfp, id_ in batch:
        f.write(f"{tfs}, {tfp}, {id_}\n")
        from_prev_list.append(tfp)
    f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poisson",
                        type=float,
                        help="Poisson arrival distribution parameter >= 0.0")
    parser.add_argument("zipf",
                        type=float,
                        help="Zipf popularity distribution parameter >= 0.0")
    parser.add_argument("disappear",
                        type=float,
                        help="disappear time Poisson distribution parameter >= 0.0")
    parser.add_argument("reappear",
                        type=float,
                        help="reappear time Poisson distribution parameter >= 0.0")
    parser.add_argument("number",
                        type=int,
                        help="number of items in trace")
    parser.add_argument("unique",
                        type=int,
                        help="unique items number > 0")
    parser.add_argument("-o",
                        "--output",
                        help="output file name. \"trace_mixed.csv\" otherwise",
                        default="trace_mixed.csv",
                        type=str)
    parser.add_argument("-b",
                        "--batch",
                        help="size of batch to be written to file in one time",
                        default=100_000,
                        type=int)
    parser.add_argument("-s",
                        "--skip",
                        help="the number of items to skip at the beginning to bring generator into stable state",
                        default=1_000_000,
                        type=int)
    args = parser.parse_args()

    with open(args.output, 'w') as f:
        generator = DisappearingPoissonZipfGenerator(args.unique,
                                                     args.poisson,
                                                     args.zipf,
                                                     0,
                                                     args.disappear,
                                                     args.reappear,
                                                     True)
        n = args.number

        with tqdm(total=args.skip, desc="Skipping first items", unit="item(s)") as pbar:
            while args.skip > args.batch:
                generator.next_n_items(args.batch)
                pbar.update(args.batch)
                args.skip -= args.batch
            if args.skip > 0:
                generator.next_n_items(args.skip)
                pbar.update(args.skip)

        with tqdm(total=n, desc="Generating items", unit="item(s)") as pbar:
            while n > args.batch:
                batch = generator.next_n_items(args.batch)
                write_batch(f, batch)
                n -= args.batch
                pbar.update(args.batch)

            if n > 0:
                batch = generator.next_n_items(n)
                write_batch(f, batch)
                pbar.update(n)

        global from_prev_list
        print("Average arrival time: {0}".format(np.mean(from_prev_list)))


if __name__ == "__main__":
    main()
