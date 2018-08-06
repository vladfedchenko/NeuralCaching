import pandas as pd
from tqdm import tqdm

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

L = list(sorted(L, reverse=True, key=lambda x: x[1]))[:300000]

L = set([x[0] for x in L])

# print(len(df))

# df = df[df.iloc[:,2].isin(L)]

# print(len(df))

with open("data/requests_300000.csv", "w") as f:
    with open("data/requests.csv", 'r') as trace:
        for i, row in tqdm(enumerate(trace), desc="Running trace", total=417882879):
            id_ = int(row.split(',')[2])
            if id_ in L:
                f.write(row)
