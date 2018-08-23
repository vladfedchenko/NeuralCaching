import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

with open("data/real_2_item_pop.csv", "r") as f:
    lines = [int(x.split(",")[1]) for x in f.readlines()]

    fig = plt.figure(1, figsize=(5, 5))
    fig.suptitle("Item popularity", fontsize=14)

    plt.hist(lines, bins=100)

    plt.xlabel("Number of requests", fontsize=14)

    plt.savefig("item_popularity_real_2.png")
