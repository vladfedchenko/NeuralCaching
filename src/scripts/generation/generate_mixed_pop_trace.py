"""
This script generates a synthetic trace of 2 populations.
First one has Poisson arrivals and Zipf popularity.
Second one has the same arrivals and popularity but objects randomly disappear and appear with Poisson distribution.
The file is generated in CSV format.
"""
import argparse
from data.generation import MixedPopulationGenerator, PoissonZipfGenerator, DisappearingPoissonZipfGenerator
from tqdm import tqdm


def write_batch(f, batch):
    for tfs, tfp, id_ in batch:
        f.write(f"{tfs}, {tfp}, {id_}\n")


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
    parser.add_argument("-pd",
                        "--poisson_dis",
                        help="""Poisson arrival distribution parameter >= 0.0 for disappearing population.
                        If not specified - the same as other population""",
                        type=int)
    parser.add_argument("-zd",
                        "--zipf_dis",
                        help="""Zipf popularity distribution parameter >= 0.0 for disappearing population.
                        If not specified - the same as other population""",
                        type=int)
    parser.add_argument("-ud",
                        "--unique_dis",
                        type=int,
                        help="""unique items number of items in disappearing population > 0.
                        If not specified - the same as other population""")
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
    args = parser.parse_args()

    if args.poisson_dis is None:
        args.poisson_dis = args.poisson

    if args.zipf_dis is None:
        args.zipf_dis = args.zipf

    if args.unique_dis is None:
        args.unique_dis = args.unique

    with open(args.output, 'w') as f:
        generator = MixedPopulationGenerator(PoissonZipfGenerator(args.unique,
                                                                  args.poisson,
                                                                  args.zipf,
                                                                  0),
                                             DisappearingPoissonZipfGenerator(args.unique_dis,
                                                                              args.poisson_dis,
                                                                              args.zipf_dis,
                                                                              args.unique,
                                                                              args.disappear,
                                                                              args.reappear,))
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
