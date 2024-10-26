import os
import re
import shap
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from model_training.data_preparer import DataPreparer
from utils.support_functions import smape, mape, rmse

tfd = tfp.distributions

class TestEvaluator:
    def __init__(self, model_filename, filename, history_length, prediction_length, input_type, model_type, target_column, data_preparer, probabilistic, results_folder, history_folder):
        self.model_filename = model_filename
        self.filename = filename
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.input_type = input_type
        self.model_type = model_type
        self.target_column = target_column
        self.data_preparer = data_preparer
        self.model = None
        self.probabilistic = probabilistic
        self.results_folder = results_folder
        self.history_folder = history_folder

    def negative_log_likelihood(self, y_true, y_pred):
        # Assuming y_pred contains mean and log variance of predictions
        mean, log_variance = tf.split(y_pred, 2, axis=-1)
        variance = tf.exp(log_variance)
        return tf.reduce_mean(0.5 * tf.log(variance) + 0.5 * (y_true - mean) ** 2 / variance)

    def log_likelihood(self, y_true, mean, variance):
        # Assume y_pred includes both mean and log_variance
        dist = tfd.Normal(loc=mean, scale=tf.sqrt(variance))
        return tf.reduce_mean(dist.log_prob(y_true))

    def load_model(self):
        custom_objects = {'smape': smape}
        if self.probabilistic:
            custom_objects['negative_log_likelihood'] = self.negative_log_likelihood
        self.model = tf.keras.models.load_model(self.model_filename, custom_objects=custom_objects)
        print(f"Loaded model from: {self.model_filename}")

    def prepare_test_data(self):
        _, _, _, _, X_test, Y_test = self.data_preparer.split_data()
        return X_test, Y_test

    def prepare_train_data(self):
        X_train, Y_train, _, _, _, _ = self.data_preparer.split_data()
        return X_train, Y_train

    def return_model_output(self):
        self.load_model()
        X_test, Y_test = self.prepare_test_data()
        Y_pred = self.model.predict(X_test)

        if self.probabilistic:
            distribution = self.model(X_test)
            mean = distribution.mean()
            variance = distribution.variance()
            mean, variance = self.data_preparer.unnormalize(mean.numpy().reshape(-1, 1), variance.numpy().reshape(-1, 1))
            Y_pred_actual = mean.flatten()
            variance_actual = variance.flatten()
        else:
            Y_pred_actual = Y_pred.flatten()
            Y_pred_actual = self.data_preparer.unnormalize(Y_pred_actual.reshape(-1, 1)).flatten()

        Y_test_actual = self.data_preparer.unnormalize(Y_test.reshape(-1, 1)).flatten()

        return Y_test_actual, Y_pred_actual

    def return_naive_output(self):
        X_test, Y_test = self.prepare_test_data()
        full_data = self.data_preparer.return_data()

        frequency = 'H'
        start_time = pd.Timestamp('2023-01-01 00:00:00')

        full_data['Hour'] = full_data.index.hour
        full_data['DayOfWeek'] = full_data.index.dayofweek

        # Calculate averages
        averages = full_data.groupby(['DayOfWeek', 'Hour'])[self.target_column].mean()

        # Generate time index for test data
        time_index = pd.date_range(start=start_time, periods=len(Y_test), freq=frequency)

        # Create a DataFrame for test data using the time index
        test_data = pd.DataFrame(data=Y_test, index=time_index, columns=[self.target_column])
        test_data['Hour'] = test_data.index.hour
        test_data['DayOfWeek'] = test_data.index.dayofweek
        test_data['Predicted'] = test_data.apply(lambda row: averages.get((row['DayOfWeek'], row['Hour']), np.nan), axis=1)

        # Fill missing predictions
        if test_data['Predicted'].isna().any():
            fill_value = averages.mean()
            test_data['Predicted'].fillna(fill_value, inplace=True)

        predicted = test_data[['Predicted']].values

        return predicted.flatten()

    def evaluate_model(self):
        self.load_model()
        X_test, Y_test = self.prepare_test_data()
        Y_pred = self.model.predict(X_test)

        if self.probabilistic:
            distribution = self.model(X_test)
            mean = distribution.mean()
            variance = distribution.variance()
            mean, variance = self.data_preparer.unnormalize(mean.numpy().reshape(-1, 1), variance.numpy().reshape(-1, 1))
            Y_pred_actual = mean.flatten()
            variance_actual = variance.flatten()
        else:
            Y_pred_actual = Y_pred.flatten()
            Y_pred_actual = self.data_preparer.unnormalize(Y_pred_actual.reshape(-1, 1)).flatten()

        Y_test_actual = self.data_preparer.unnormalize(Y_test.reshape(-1, 1)).flatten()
        self.plot_results(Y_test_actual, Y_pred_actual)
        self.plot_training_history()

        return self.calculate_metrics(Y_test_actual, Y_pred_actual, variance_actual if self.probabilistic else None)

    def calculate_metrics(self, y_true, y_pred, variance):
        # Calculate metrics
        rmse_value = rmse(y_true, y_pred)
        mape_value = mape(y_true, y_pred)
        smape_tensor = smape(tf.convert_to_tensor(y_true, dtype=tf.float32), tf.convert_to_tensor(y_pred, dtype=tf.float32))
        smape_value = float(smape_tensor.numpy())

        if self.probabilistic:
            log_likelihood_value = self.log_likelihood(y_true, y_pred, variance).numpy()
            print(f"Log-Likelihood: {log_likelihood_value}")
            return (rmse_value, mape_value, smape_value, log_likelihood_value)
        else:
            return (rmse_value, mape_value, smape_value)

    def plot_results(self, actuals, predictions):

        folder_path = os.path.join(self.results_folder + self.filename, "Prediction Plots")
        os.makedirs(folder_path, exist_ok=True)

        # Calculate positive residuals
        residuals = abs(actuals - predictions)

        plt.figure(figsize=(30, 6))

        # Plot actuals and predictions
        ax = plt.gca()  # Get current axis
        ax.plot(actuals, label='Actual Values', color='blue')
        ax.plot(predictions, label='Predicted Values', color='red', linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Comparison of Actual and Predicted Values ({self.history_length}-{self.prediction_length}-{self.input_type}-{self.model_type}_{self.filename})')
        ax.legend(loc='upper left')
        ax.grid(True)
        #ax.set_ylim(21, 23)  # Set y-axis range for LHS

        # Create a second y-axis for positive residuals
        ax2 = ax.twinx()
        ax2.plot(residuals, label='Residual Error', color='green', linestyle=':')
        ax2.set_ylabel('Residuals')
        ax2.legend(loc='upper right')
        #ax2.set_ylim(0, 1)  # Set y-axis range for RHS

        # Save the plot to the specified folder
        figure_path = os.path.join(folder_path, f'{self.history_length}-{self.prediction_length}-{self.input_type}-{self.model_type}_{self.filename}_plot.pdf')
        plt.savefig(figure_path)
        plt.close()  # Close the plot explicitly after saving

    def plot_training_history(self):
        # Construct the full path to the corresponding history JSON file
        model_base_name = os.path.basename(self.model_filename)
        history_filename = model_base_name.replace('.keras', '_history.json')
        history_file_path = os.path.join(self.history_folder + self.filename, history_filename)

        history_save_path = self.results_folder + self.filename + "/History Plots"
        os.makedirs(history_save_path, exist_ok=True)

        if os.path.exists(history_file_path):
            with open(history_file_path, 'r') as file:
                history = json.load(file)

            # Extracting loss and SMAPE metrics
            loss = history.get('loss', [])
            val_loss = history.get('val_loss', [])

            # Plotting training and validation loss
            plt.figure(figsize=(12, 6))
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Save plot to the results folder
            plot_path = os.path.join(history_save_path, history_filename.replace('_history.json', f'_{self.filename}_training_plot.pdf'))
            plt.savefig(plot_path)
            plt.close()

    def evaluate_model_with_shap(self):

        self.load_model()

        # Ensure the model is loaded properly
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model is not properly configured with a predict method.")

        X_train, Y_train = self.prepare_train_data()
        X_test, Y_test = self.prepare_test_data()

        # Assuming X_train and X_test are already defined and properly formatted
        background = X_train[2500:3500]

        explainer = shap.GradientExplainer(self.model, background)

        # Initialize an empty list to collect data
        results = []

        # Loop through all instances in X_test
        for i in range(X_test.shape[0]):
            X_test_subset = X_test[i:i+1]  # Get the test instance at index i
            shap_values = explainer.shap_values(X_test_subset, nsamples=100)  # Compute SHAP values

            # Store each instance's SHAP values and the corresponding test subset
            result = {
                "index": i,
                "shap_values": shap_values,
                "X_test_subset": X_test_subset
            }
            results.append(result)

        return results

    def identify_peaks(self, actuals, predictions, threshold=0.6, prominence=0.6, peak_tolerance=10):
        peak_differences = []
        false_positives = []
        false_negatives = []
        diff_magnitudes = []
        peak_index_diffs = []

        # Identify peaks in actuals
        actual_peaks, _ = find_peaks(actuals, distance=self.prediction_length, prominence=prominence)

        # Identify peaks in predictions
        predicted_peaks, _ = find_peaks(predictions, distance=self.prediction_length, prominence=prominence)

        # Convert NumPy arrays to lists for easier manipulation
        actual_peaks = actual_peaks.tolist()
        predicted_peaks = predicted_peaks.tolist()

        # Compare peaks and calculate differences
        for actual_peak_index in actual_peaks:
            found_match = False
            for predicted_peak_index in predicted_peaks[:]:
                if abs(actual_peak_index - predicted_peak_index) <= peak_tolerance:
                    actual_peak_value = actuals[actual_peak_index]
                    predicted_peak_value = predictions[predicted_peak_index]
                    epsilon = 1e-10  # A small number to prevent division by zero
                    diff_magnitude = 100 * abs(actual_peak_value - predicted_peak_value) / ((abs(actual_peak_value) + abs(predicted_peak_value) + epsilon) / 2)

                    if diff_magnitude >= threshold:
                        peak_diff = {
                            'index': actual_peak_index,
                            'actual': actual_peak_value,
                            'predicted': predicted_peak_value,
                            'diff_magnitude': diff_magnitude
                        }
                        peak_differences.append(peak_diff)
                        diff_magnitudes.append(diff_magnitude)
                        peak_index_diffs.append(abs(actual_peak_index - predicted_peak_index))
                    found_match = True
                    predicted_peaks.remove(predicted_peak_index)
                    break

            if not found_match:
                false_negatives.append(actual_peak_index)

        false_positives = predicted_peaks

        average_diff_magnitude = sum(diff_magnitudes) / len(diff_magnitudes) if diff_magnitudes else 0
        average_peak_index_diff = sum(peak_index_diffs) / len(peak_index_diffs) if peak_index_diffs else 0

        return {
            'peak_differences': peak_differences,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'average_diff_magnitude': average_diff_magnitude,
            'average_peak_index_diff': average_peak_index_diff
        }

    def evaluate_model_peaks(self):
        self.load_model()
        X_test, Y_test = self.prepare_test_data()
        Y_pred = self.model.predict(X_test)

        if self.probabilistic:
            distribution = self.model(X_test)
            mean = distribution.mean()
            variance = distribution.variance()
            mean, variance = self.data_preparer.unnormalize(mean.numpy().reshape(-1, 1), variance.numpy().reshape(-1, 1))
            Y_pred_actual = mean.flatten()
            variance_actual = variance.flatten()
        else:
            Y_pred_actual = Y_pred.flatten()
            Y_pred_actual = self.data_preparer.unnormalize(Y_pred_actual.reshape(-1, 1)).flatten()

        Y_test_actual = self.data_preparer.unnormalize(Y_test.reshape(-1, 1)).flatten()

        peak_info = self.identify_peaks(Y_test_actual, Y_pred_actual)

        folder_path = os.path.join(self.results_folder + self.filename, "Peak Plots")
        os.makedirs(folder_path, exist_ok=True)

        plt.figure(figsize=(30, 6))

        # Plot actuals and predictions
        ax = plt.gca()  # Get current axis
        ax.plot(Y_test_actual, label='Actual Values', color='blue')
        ax.plot(Y_pred_actual, label='Predicted Values', color='red', linestyle='--')

        # Highlight peak differences
        for peak in peak_info['peak_differences']:
            ax.plot(peak['index'], Y_test_actual[peak['index']], 'ko', markersize=5)
            ax.plot(peak['index'], Y_pred_actual[peak['index']], 'ko', markersize=5)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Comparison of Actual and Predicted Values ({self.history_length}-{self.prediction_length}-{self.input_type}-{self.model_type}_{self.filename})')
        ax.legend(loc='upper left')
        ax.grid(True)

        figure_path = os.path.join(folder_path, f'{self.history_length}-{self.prediction_length}-{self.input_type}-{self.model_type}_{self.filename}_plot.pdf')
        plt.savefig(figure_path)
        plt.close()

        return peak_info
    
def extract_model_metadata(filename):
    """Extract configuration from the filename using a specific regex pattern.

    Expected format: '{history_length}-{prediction_length}-{input_type}-{model_type}.keras'
    Example: '1-1-univariate-LSTMModel.keras'
    """
    pattern = r"(\d+)-(\d+)-(univariate|multivariate)-([A-Za-z]+Model)\.keras"
    match = re.search(pattern, filename)
    if match:
        try:
            history_length = int(match.group(1))
            prediction_length = int(match.group(2))
            input_type = match.group(3)
            model_type = match.group(4)
            return history_length, prediction_length, input_type, model_type
        except ValueError as e:
            print(f"Error processing {filename}: {e}")
            raise
    else:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    
def evaluate_all_models(model_dir, data_path, filename, results_folder, history_folder, target_column, probabilistic):
    data = pd.read_csv(os.path.join(data_path, filename + '.csv'))
    results_df = pd.DataFrame(columns=['Filename', 'History Length', 'Prediction Length', 'Input Type', 'Model Type', 'RMSE', 'MAPE', 'SMAPE'])

    for weights in os.listdir(model_dir+filename):
        if weights.endswith(".keras"):
            try:
                history_length, prediction_length, input_type, model_type = extract_model_metadata(weights)
                model_path = os.path.join(model_dir+filename, weights)
                print(f"Evaluating {weights} with HL={history_length}, PL={prediction_length}, IT={input_type}, MT={model_type}")

                data_preparer_test = DataPreparer(data=data, target_column=target_column, history_length=history_length, prediction_length=prediction_length, input_type=input_type, datetime_column='Time')
                evaluator = TestEvaluator(model_path, filename, history_length, prediction_length, input_type, model_type, target_column, data_preparer_test, probabilistic, results_folder, history_folder)

                if input_type == 'multivariate':
                    # Evaluate model with SHAP
                    shap_values = evaluator.evaluate_model_with_shap()
                    shapley_save = f"{results_folder+filename}/Shapley Values/"
                    os.makedirs(shapley_save, exist_ok=True)
                    np.save(shapley_save + f"{history_length}-{prediction_length}-{input_type}-{model_type}_shap_values.npy", shap_values)

                if probabilistic:
                    rmse, mape, smape, loglike = evaluator.evaluate_model()

                    new_row = pd.DataFrame([{
                        'Filename': filename,
                        'History Length': history_length,
                        'Prediction Length': prediction_length,
                        'Input Type': input_type,
                        'Model Type': model_type,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'SMAPE': smape,
                        'Log-Likelihood': loglike
                    }])

                else:
                    rmse, mape, smape = evaluator.evaluate_model()

                    new_row = pd.DataFrame([{
                        'Filename': filename,
                        'History Length': history_length,
                        'Prediction Length': prediction_length,
                        'Input Type': input_type,
                        'Model Type': model_type,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'SMAPE': smape
                    }])

                results_df = pd.concat([results_df, new_row], ignore_index=True)

            except Exception as e:  # Broad exception to catch any error
                print(f"Error processing {filename}: {e}")

    return results_df