import os
import glob
import numpy as np
from tqdm.notebook import tqdm


def load_abm_data(data_path, config):
    # data = []
    # for path, subdirs, files in os.walk(data_path):
    #     for name in files:
    #         if (name == "daily_city_grid.npy"):
    #             data.append(np.load(os.path.join(path, name)))

    # return data
    sub_grid_list = sorted(glob.glob("{}/*/*.npy".format(data_path)))
    sub_grid_R0_list = [float(sim_path.split('/')[-2].split('_')[2]) for sim_path in sub_grid_list]
    print("Length =", len(sub_grid_R0_list))
    print("sub_grid_R0_list", sub_grid_R0_list)
    sub_grid_list = np.array(sub_grid_list)
    sub_grid_R0_list = np.array(sub_grid_R0_list)

    indices = np.arange(len(sub_grid_list))
    np.random.shuffle(indices)

    sub_grid_list = sub_grid_list[indices]
    sub_grid_R0_list = sub_grid_R0_list[indices]
    X_train = []
    y_train = []

    X_val = []
    y_val = []

    X_test = []
    y_test = []
    X_test_R0s = []

    num_train_sims = round(config.train_fraction*len(sub_grid_list))
    num_val_sims = round((config.train_fraction+config.val_fraction)*len(sub_grid_list))

    # for sdx, sim_path in tqdm(enumerate(sims), total=len(sims)):
    for sdx in tqdm(range(len(sub_grid_list))):
        # daily_city_grid = np.load(sim_path)
        daily_city_grid = np.load(sub_grid_list[sdx])
        sim_days = daily_city_grid.shape[0]
        R0 = sub_grid_R0_list[sdx]

        xdx = 0
        for ydx in range(config.context_len, sim_days):
            
            X = daily_city_grid[xdx:ydx, :, :, :].copy()    
            y = daily_city_grid[ydx, :, :, :config.num_feat_cols].copy()

            if (sdx+1 <= num_train_sims):
                X_train.append(X)
                y_train.append(y)

            elif (sdx+1 <= num_val_sims):
                X_val.append(X)
                y_val.append(y)
            
            else:
                X_test.append(X)
                y_test.append(y)

            xdx += 1
        
        if (sdx+1 > num_val_sims):
            X_test_R0s.append(R0)

    X_train = [ x/int(config.block_count) for x in X_train]

    X_train[:, :, :, :, -1] = X_train[:, :, :, :, -1]*config.block_count
    y_train = np.array(y_train)/config.block_count

    X_val = np.array(X_val)/config.block_count
    X_val[:, :, :, :, -1] = X_val[:, :, :, :, -1]*config.block_count
    y_val = np.array(y_val)/config.block_count
    X_test = np.array(X_test)/config.block_count
    X_test[:, :, :, :, -1] = X_test[:, :, :, :, -1]*config.block_count
    y_test = np.array(y_test)/config.block_count

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_test_R0s

if __name__ == "__main__":
    data = load_abm_data("/content/drive/MyDrive/Emulator_Phase_2/Simulator_v1.7/results/v1.7_R0_Experiments/")
    print(data[0].shape)
