import math
import numpy as np
import pandas as pd
import gc

class DataLoader:
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, filename, cols, type):
        model_dataframe = pd.read_pickle(filename)
        model_dataframe.fillna(0, inplace=True)
        if type == 'trade':
            self.data_train = model_dataframe[model_dataframe['Trade Name']<=11][cols].values
            self.data_test = model_dataframe[model_dataframe['Trade Name']>11][cols].values
        else: #type == 'time'
            self.data_train = model_dataframe[model_dataframe['Value Date']< '2024-01-1'][cols].values
            self.data_test = model_dataframe[model_dataframe['Value Date']>= '2024-01-1'][cols].values
        del model_dataframe; gc.collect()
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def get_test_data(self, seq_len, normalise):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        """
        data_windows = np.array([self.data_test[i:i+seq_len] for i in range(self.len_test - seq_len)])
        if normalise:
            data_windows = self._normalise_windows(data_windows)
        
        x_test = data_windows[:, :-1]
        y_test = data_windows[:, -1, 0]
        return x_test, y_test

    def get_train_data(self, seq_len, normalise):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        data_windows = np.array([self._next_window(i, seq_len, normalise) for i in range(self.len_train - seq_len)])
        x_train = data_windows[:, :-1]
        y_train = data_windows[:, -1, 0]
        return x_train, y_train

    def generate_train_batch(self, seq_len, batch_size, normalise, indices):
        """Yield a generator of training data from filename on given list of cols split for train/test"""
        while True:
            np.random.shuffle(indices)  # Ensure indices are shuffled each time
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                data_windows = [self._next_window(i, seq_len, normalise) for i in batch_indices if i < (self.len_train - seq_len)]
                if data_windows:
                    x_batch = np.array([window[:-1] for window in data_windows])
                    y_batch = np.array([window[-1, 0] for window in data_windows])
                    yield x_batch, y_batch

    def _next_window(self, i, seq_len, normalise):
        """Generates the next data window from the given index location i"""
        window = self.data_train[i:i+seq_len]
        if normalise:
            window = self._normalise_windows(window[None, ...])[0]
        return window

    def _normalise_windows(self, window_data):
        """Normalise window with a base value of zero using vectorized operations"""
        base = window_data[:, 0, :]  # shape will be [num_windows, num_features]
        normalised_data = np.where(base[:, None, :] != 0,
                                   (window_data - base[:, None, :]) / base[:, None, :],
                                   0)  # If the base value is zero, the normalized result is also zero
        return normalised_data