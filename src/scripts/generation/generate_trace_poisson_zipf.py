"""
This script generates a synthetic trace of population with Poisson arrivals and Zipf popularity.
The file is generated in CSV format.
"""
import argparse
from data.generation import PoissonZipfGenerator
from tqdm import tqdm


def write_batch(f, batch):
    for tfs, tfp, id_ in batch:
        f.write(f"{tfs}, {tfp}, {id_}\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poisson",
                        type=float,
                        help="Poisson arrival distribution parameter >= 0.0")
    parser.add_argument("zipf",
                        type=float,
                        help="Zipf popularity distribution parameter >= 0.0")
    parser.add_argument("number",
                        type=int,
                        help="number of items in trace")
    parser.add_argument("unique",
                        type=int,
                        help="unique items number > 0")
    parser.add_argument("-o",
                        "--output",
                        help="output file name. \"trace.csv\" otherwise",
                        default="trace.csv",
                        type=str)
    parser.add_argument("-b",
                        "--batch",
                        help="size of batch to be requested from generator in one time",
                        default=100_000,
                        type=int)
    args = parser.parse_args()

    with open(args.output, 'w') as f:
        generator = PoissonZipfGenerator(args.unique, args.poisson, args.zipf)
        n = args.number

        with tqdm(total=n) as pbar:
            while n > args.batch:
                batch = generator.next_n_items(args.batch)
                write_batch(f, batch)
                n -= args.batch
                pbar.update(args.batch)

            if n > 0:
                batch = generator.next_n_items(n)
                write_batch(f, batch)
                pbar.update(n)


if __name__ == "__main__":
    main()
