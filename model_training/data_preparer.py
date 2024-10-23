import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow_probability as tfp

tfd = tfp.distributions

class DataPreparer:
    def __init__(self, data, target_column, history_length, prediction_length, train_split=0.6, val_split=0.2, input_type='multivariate', include_target_in_features=True, probabilistic=False, datetime_column=None):
        self.data = data
        self.target_column = target_column
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.train_split = train_split
        self.val_split = val_split
        self.input_type = input_type
        self.probabilistic = probabilistic
        self.include_target_in_features = include_target_in_features
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.datetime_column = datetime_column

        if datetime_column and not isinstance(self.data.index, pd.DatetimeIndex):
            self.convert_to_datetime_index(datetime_column)

        self.datetime_indices = {'train_start': None, 'val_start': None, 'test_start': None}

        # Preprocess and split the data
        self.data_prepared, self.target_data = self.initialize_and_prepare_data()
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self.split_data()

    def convert_to_datetime_index(self, datetime_column):
        """Converts the specified column to a DatetimeIndex."""
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data.set_index(datetime_column, inplace=True)

    def initialize_and_prepare_data(self):
        if self.input_type == 'univariate':
            features = self.data[[self.target_column]]
        else:
            features = self.data.select_dtypes(include=[np.number])
            if not self.include_target_in_features:
                features = features.drop(columns=[self.target_column], errors='ignore')

        # Fit and transform features
        features_scaled = self.scaler.fit_transform(features)
        # Fit and transform the target data
        target_scaled = self.target_scaler.fit_transform(self.data[[self.target_column]])

        return features_scaled, target_scaled

    def prepare_sequences(self, features, target):
        X, y = [], []
        total_samples = len(features) - self.history_length - self.prediction_length + 1  # Ensure we have enough data for both history and prediction

        for i in range(total_samples):
            end_ix = i + self.history_length
            if self.probabilistic:
                out_end_ix = end_ix + self.prediction_length
                seq_y = target[end_ix:out_end_ix]
            else:
                out_end_ix = end_ix + self.prediction_length - 1
                seq_y = target[out_end_ix]

            if out_end_ix > len(features):
                continue

            seq_x = features[i:end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y).reshape(-1, 1 if not self.probabilistic else self.prediction_length)

    def split_data(self):
        X, y = self.prepare_sequences(self.data_prepared, self.target_data.flatten())
        num_train = int(len(X) * self.train_split)
        num_val = int(len(X) * self.val_split)
        num_test = len(X) - num_train - num_val

        if self.datetime_column:
            # Record the datetime index starts for each split
            self.datetime_indices['train_start'] = self.data.index[self.history_length]
            self.datetime_indices['val_start'] = self.data.index[self.history_length + num_train]
            self.datetime_indices['test_start'] = self.data.index[self.history_length + num_train + num_val]

        return X[:num_train], y[:num_train], X[num_train:num_train + num_val], y[num_train:num_train + num_val], X[num_train + num_val:], y[num_train + num_val:]

    def transform_data(self, new_data):
        if self.input_type == 'univariate':
            new_data = new_data[[self.target_column]]
        else:
            new_data = new_data.select_dtypes(include=[np.number])
            if not self.include_target_in_features:
                new_data = new_data.drop(columns=[self.target_column], errors='ignore')
        return self.scaler.transform(new_data)  # Use fitted scaler to transform new data

    def unnormalize(self, y_pred, variance=None):
        # Unnormalize predictions to original scale
        unnormalized_predictions = self.target_scaler.inverse_transform(y_pred)

        if variance is not None:
            # Adjust variance to match the scale of the original data
            # We assume that self.target_scaler.data_range_ returns the (max - min) of the target data used during fitting
            scale_factor = self.target_scaler.data_range_[0] ** 2
            adjusted_variance = variance * scale_factor
            return unnormalized_predictions, adjusted_variance

        return unnormalized_predictions

    def get_split_start_dates(self):
        """ Return the starting datetime for each dataset split. """
        return self.datetime_indices

    def get_column_names(self):
      return self.data.columns

    def return_data(self):
      return self.data