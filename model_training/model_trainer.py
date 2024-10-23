import os
import json
import click
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.wandb_keras import WandbCallback
from utils.support_functions import load_best_performances, update_best_performance, smape, SMAPECallback
from model_training.data_preparer import DataPreparer
from model_training.model_creator import LSTMModel, CNNModel, CNNLSTMModel

class ModelTrainer:
    def __init__(self, model, X_train, Y_train, X_val, Y_val, data_preparer, model_name, save_path, history_path, probabilistic, track_file):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.data_preparer = data_preparer
        self.model_name = model_name
        self.save_path = save_path
        self.history_path = history_path
        self.probabilistic = probabilistic
        self.track_file = track_file

    def negative_log_likelihood(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)

    def train(self):
        config = wandb.config

        # Select the optimizer based on configuration
        if config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        elif config.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
        elif config.optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=config.learning_rate)
        elif config.optimizer == 'lion':
            optimizer = tf.keras.optimizers.Lion(learning_rate=config.learning_rate)
        elif config.optimizer == 'ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=config.learning_rate)
        elif config.optimizer == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
        elif config.optimizer == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=config.learning_rate)
        elif config.optimizer == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate)
        elif config.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer}")

        # Compile the model with the selected optimizer
        loss = 'mean_squared_error' if not self.probabilistic else self.negative_log_likelihood
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[smape])

        # Include the SMAPE callback with validation data and data preparer for unnormalization
        smape_callback = SMAPECallback((self.X_val, self.Y_val), self.data_preparer)

        model_key = f"{self.model_name}"
        best_performances = load_best_performances(self.track_file)
        best_val_loss = best_performances.get(model_key, {}).get('val_loss', float('inf'))
        best_model_path = ""

        history = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val),
            epochs=config.epochs, batch_size=config.batch_size,
            callbacks=[WandbCallback(), smape_callback])  # Add WandbCallback to log other metrics and training process

        final_val_loss = history.history['val_loss'][-1]
        if final_val_loss < best_val_loss:
            best_model_path = os.path.join(self.save_path, f"{model_key}.keras")
            self.model.save(best_model_path, save_format='tf')
            update_best_performance(self.track_file, model_key, final_val_loss, best_model_path)

            history_file_path = os.path.join(self.history_path, f"{model_key}_history.json")
            with open(history_file_path, 'w') as hist_file:
                json.dump(history.history, hist_file)

        wandb.finish()

        return best_model_path if best_model_path else "No improvement"

# Main execution function
def create_sweep_config(target_column, history_length, prediction_length, model_type, input_type, save_path, data_path, history_path, probabilistic, track_file):
    return {
        'method': 'bayes',  # Use Bayesian optimization
        'name': f"{history_length}-{prediction_length}-{input_type}-{model_type}",
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'target_column': {'value': target_column},
            'data_path': {'value': data_path},
            'history_path': {'value': history_path},
            'track_file': {'value': track_file},
            'save_path': {'value': save_path},
            'probabilistic': {'value': probabilistic},
            'history_length': {'value': history_length},
            'prediction_length': {'value': prediction_length},
            'input_type': {'value': input_type},
            'model_type': {'value': model_type},
            'lstm_units': {'values': [1, 2, 4, 8, 16, 32, 64, 128]},
            'learning_rate': {'min': 0.00001, 'max': 0.1},
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'value': 50},
            'filters_1': {'values': [32, 64, 128]},
            'filters_2': {'values': [32, 64, 128]},
            'kernel_size_1': {'values': [3, 5, 7]},
            'kernel_size_2': {'values': [3, 5, 7]},
            'cnn_kernel_size_1': {'values': [1, 2, 3]},
            'cnn_kernel_size_2': {'values': [1, 2, 3]},
            'cnn_layers': {'values': [1, 2, 3, 4, 5]},
            'lstm_layers': {'values': [1, 2, 3, 4, 5]},
            'dense_units': {'values': [64, 128, 256]},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'lion', 'ftrl', 'nadam', 'adamax', 'adagrad', 'rmsprop']},
            'activation': {'values': ['elu', 'selu', 'gelu', 'leaky_relu', 'relu', 'tanh']},
            'dilation_rate_1': {'values': [1, 2, 4]},
            'dilation_rate_2': {'values': [1, 2, 4]}
        }
    }

def model_training():
    try:
        # Initialize wandb session
        run = wandb.init(reinit=True)
        config = wandb.config

        # Load data and prepare models as before
        data = pd.read_csv(config.data_path)
        preparer = DataPreparer(data, config.target_column, history_length=config.history_length, prediction_length=config.prediction_length, input_type=config.input_type, probabilistic=config.probabilistic)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = preparer.split_data()

        model_name = f"{config.history_length}-{config.prediction_length}-{config.input_type}-{config.model_type}"

        # Dynamically select the model based on the sweep configuration
        if config.model_type == 'LSTMModel':
            model = LSTMModel((None, X_train.shape[1], X_train.shape[2]), config, config.probabilistic)
        elif config.model_type == 'CNNModel':
            model = CNNModel((None, X_train.shape[1], X_train.shape[2]), config, config.probabilistic)
        elif config.model_type == 'CNNLSTMModel':
            model = CNNLSTMModel((None, X_train.shape[1], X_train.shape[2]), config, config.probabilistic)
        else:
            raise ValueError("Unsupported model type")

        trainer = ModelTrainer(model.model, X_train, Y_train, X_val, Y_val, preparer, model_name, config.save_path, config.history_path, config.probabilistic, config.track_file)
        best_model_path = trainer.train()

        # Log the best model path to wandb for this run
        wandb.log({'best_model_path': best_model_path})

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Finish the wandb session
        wandb.finish()