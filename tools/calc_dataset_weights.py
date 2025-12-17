import sys
import numpy as np
import json

"""
Format: dataset prefix as key, with a tuple of (#tokens, #epochs in training, #full path of file)
"""
# example_data_banks = {
#     "prefix1": (25085726880, 0.88, "path1"),
#     "prefix2": (28163630944, 3.15, "path2"),
#     "prefix3": (33085507456, 1.14, "path3"),
#     "prefix4": (270594169952, 4.0, "path4"),
#     "prefix5": (197574638208, 6.0, "path5"),
# }

def calc_dataset_weights(data_banks):
    total_training_tokens = sum([v[0] * v[1] for k, v in data_banks.items()])

    command = []
    for k, v in data_banks.items():
        path = v[-1]
        # assert os.path.exists(path)
        weight = v[0] * v[1] / total_training_tokens * 100
        # print(f"{k}: {weight:.2f}%")
        command.append(f"{weight:.10f} {path}")

    # print(f"Total train tokens: {total_training_tokens / 1e12:.2f}T")
    # print("=========================================")
    command = " ".join(command)
    print(command)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python calc_dataset_weights.py <data_banks.json>")
        sys.exit(1)
    
    data_banks = json.load(open(sys.argv[1]))
    calc_dataset_weights(data_banks)
