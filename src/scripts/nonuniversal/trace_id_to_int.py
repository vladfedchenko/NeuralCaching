input_df = "data/real_trace_2_small.csv"
output_df = "data/real_trace_2_small_id_fix.csv"

id_map = {}
i = 1
with open(input_df, "r") as inp:
    with open(output_df, "w") as outp:
        for line in inp:
            arr = line.split(",")
            if arr[2] not in id_map:
                id_map[arr[2]] = i
                i += 1

            outp.write("{}, {}, {}\n".format(arr[0], arr[1], id_map[arr[2]]))
