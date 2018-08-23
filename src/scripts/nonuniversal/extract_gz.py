import os
import subprocess
import io
from natsort import natsorted
from tqdm import tqdm
import numpy as np

dirname = "data/log_share"

fl = list(natsorted(os.listdir(dirname)))

req_map = {}
for fn in tqdm(fl):
    if not fn.endswith(".gz"):
        continue
    fullfn = os.path.join(dirname, fn)

    p = subprocess.Popen(["zcat", fullfn], stdout=subprocess.PIPE)
    fh = io.BytesIO(p.communicate()[0])
    assert p.returncode == 0

    for line in fh:
        row = line.split()
        id_ = row[2].decode("utf-8")
        if id_ in req_map:
            req_map[id_] += 1
        else:
            req_map[id_] = 1

L = list(req_map.items())

with open("data/real_2_item_pop.csv", 'w') as pop_file:
    for id_, pop in L:
        pop_file.write("{},{}\n".format(id_, pop))

