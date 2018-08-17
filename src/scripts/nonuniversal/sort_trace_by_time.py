import pandas as pd

input_file = "data/real_trace_2.csv"
output_file = "data/real_trace_2_sorted.csv"

df = pd.read_csv(input_file, header=None, names=None, index_col=None)
df.sort_values(0, inplace=True)

df.to_csv(output_file, header=False, index=False)
