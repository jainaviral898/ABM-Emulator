import os
import glob
import numpy as np


def load_abm_data(data_path):
    data = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if (name == "daily_city_grid.npy"):
                data.append(np.load(os.path.join(path, name)))

    return data


if __name__ == "__main__":
    data = load_abm_data("data-dir/Simulator_v1.7/results/v1.7_R0_Experiments/")
    print(data[0].shape)
