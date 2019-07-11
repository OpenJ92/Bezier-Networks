import numpy as np
import pandas as pd

def save_data_to_directory(data, root_dir):
    unique_id = data['series_id'].unique()
    np.save(root_dir + '/' + 'seriesID.npy', unique_id)
    import pdb;pdb.set_trace()
    for id in unique_id:
        series = data[data['series_id'] == id]
        series = series.values
        np.save(root_dir + '/' + str(id) + '.npy' , series)

if __name__ == "__main__":
    X_train = pd.read_csv('X_train/X_train.csv')
    y_train = pd.read_csv('y_train/y_train.csv')
    X_test = pd.read_csv('X_test/X_test.csv')
    
    save_data_to_directory(X_train, 'train_X')
    save_data_to_directory(y_train, 'train_y')
    save_data_to_directory(X_test, 'test_X')
    