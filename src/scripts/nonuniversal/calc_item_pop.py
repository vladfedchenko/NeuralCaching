import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

req_map = {}
with open("data/requests.csv", 'r') as trace:
    for i, row in tqdm(enumerate(trace), desc="Running trace", total=417882879):
        row = row.split(',')
        id_ = int(row[2])
        if id_ in req_map:
            req_map[id_] += 1
        else:
            req_map[id_] = 1

L = list(req_map.items())

with open("data/requests_item_pop.csv", 'w') as pop_file:
    for id_, pop in L:
        pop_file.write("{},{}\n".format(id_, pop))



