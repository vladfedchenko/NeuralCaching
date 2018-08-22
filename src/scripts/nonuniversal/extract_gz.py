import os
import subprocess
import io
from natsort import natsorted
from tqdm import tqdm
import numpy as np

dirname = "data/log_share"

fl = list(natsorted(os.listdir(dirname)))

total_write = 100_000_000

# with tqdm.tqdm(total=total_write) as prog_bar:
#     with open("data/real_trace_2.csv", "w") as f:

min_size = 10**10
max_size = 0

size_sum = 0
all_size = []

i = 0
for fn in tqdm(fl):
    if not fn.endswith(".gz"):
        continue
    fullfn = os.path.join(dirname, fn)

    p = subprocess.Popen(["zcat", fullfn], stdout=subprocess.PIPE)
    fh = io.BytesIO(p.communicate()[0])
    assert p.returncode == 0

    for line in fh:
        arr = line.split()
        size = int(arr[1].decode("utf-8"))

        if min_size > size:
            min_size = size

        if max_size < size:
            max_size = size

        size_sum += size
        i += 1
        if i == 1_000_000:
            i = 0
            size_sum /= 1_000_000
            all_size.append(size_sum)

            size_sum = 0
    #     f.write("{}, {}, {}\n".format(arr[0].decode("utf-8"), arr[1].decode("utf-8"), arr[2].decode("utf-8")))
    #     prog_bar.update(1)
    #     i += 1
    #     if i == total_write:
    #         break
    #
    # if i == total_write:
    #     break

mean = np.mean(all_size)

print("Min: {}, Max: {}, Mean: {}".format(min_size, max_size, mean))