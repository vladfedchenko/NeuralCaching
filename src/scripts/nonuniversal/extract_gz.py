import os
import subprocess
import io
from natsort import natsorted
import tqdm

dirname = "data/log_share"

fl = natsorted(os.listdir(dirname))

total_write = 100_000_000
i = 0
with tqdm.tqdm(total=total_write) as prog_bar:
    with open("data/real_trace_2.csv", "w") as f:
        for fn in fl:
            if not fn.endswith(".gz"):
                continue
            fullfn = os.path.join(dirname, fn)

            p = subprocess.Popen(["zcat", fullfn], stdout=subprocess.PIPE)
            fh = io.BytesIO(p.communicate()[0])
            assert p.returncode == 0

            for line in fh:
                arr = line.split()
                f.write("{}, {}, {}\n".format(arr[0].decode("utf-8"), 1, arr[1].decode("utf-8")))
                prog_bar.update(1)
                i += 1
                if i == total_write:
                    break

            if i == total_write:
                break
