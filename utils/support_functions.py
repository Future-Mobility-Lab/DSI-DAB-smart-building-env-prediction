import os
import json
import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error

def load_best_performances(track_file):
    if os.path.exists(track_file):
        with open(track_file, 'r') as file:
            return json.load(file)
    return {}

def update_best_performance(track_file, model_key, val_loss, model_path):
    best_performances = load_best_performances(track_file)
    if model_key not in best_performances or val_loss < best_performances[model_key]['val_loss']:
        best_performances[model_key] = {'val_loss': val_loss, 'model_path': model_path}
        with open(track_file, 'w') as file:
            json.dump(best_performances, file)

def smape(y_true, y_pred):
    # Convert y_true and y_pred to tensors of the same type, float32
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0
    diff = tf.abs(y_true - y_pred) / denominator
    diff = tf.where(tf.math.is_nan(diff), tf.zeros_like(diff), diff)  # Handle NaNs possibly caused by zero division
    return 100.0 * tf.reduce_mean(diff)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class SMAPECallback(Callback):
    def __init__(self, val_data, data_preparer):
        super().__init__()
        self.X_val, self.Y_val = val_data
        self.data_preparer = data_preparer

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.X_val)
        val_pred_unnorm = self.data_preparer.unnormalize(val_pred)
        y_true_unnorm = self.data_preparer.unnormalize(self.Y_val)

        smape_value = self.calculate_smape(y_true_unnorm, val_pred_unnorm)
        logs['val_smape'] = float(smape_value)  # Ensure this is a float, not np.float32

        # Log SMAPE to wandb
        wandb.log({'val_smape': smape_value}, commit=False)

    def calculate_smape(self, y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))