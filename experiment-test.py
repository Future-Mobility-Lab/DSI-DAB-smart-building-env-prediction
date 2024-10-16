import os
import pandas as pd
from model_testing.test_evaluator import evaluate_all_models

# Configuration and data loading
folder_path = '/workspace'
target_column = "t"
data_path = f'{folder_path}/Processed Datasets/'
model_dir = f'{folder_path}/DSI DAB Input Type/{target_column}/Model Weights/'
history_folder = f'{folder_path}/DSI DAB Input Type/{target_column}/Model Training History/'
results_folder = f'{folder_path}/Evaluation/DSI DAB Input Type/{target_column}/'
probabilistic = False

results = pd.DataFrame(columns=['Filename', 'History Length', 'Prediction Length', 'Input Type', 'Model Type', 'RMSE', 'MAPE', 'SMAPE'])

# Evaluate all models
for filename in os.listdir(model_dir):
    # Construct the full file path
    output = evaluate_all_models(model_dir, data_path, filename, results_folder, history_folder, target_column, probabilistic)
    results = pd.concat([results, output], ignore_index=True)

# Save the results to a CSV file
results.to_csv(f'{results_folder}/results.csv', index=False)