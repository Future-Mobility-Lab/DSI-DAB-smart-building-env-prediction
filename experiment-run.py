import os
import wandb
from model_training.model_trainer import create_sweep_config, model_training

history_lengths = list(range(1, 13))  # From 1 to 12
prediction_lengths = list(range(1, 13))  # From 1 to 12
model_types = ['CNNModel', 'LSTMModel', 'CNNLSTMModel']
input_types = ['univariate', 'multivariate']
target_columns = ['t']
folder_path = '/workspace'
data_path = f'{folder_path}/Processed Datasets/'
probabilistic = False
count = 50
project_name =  "HASS_DSI_experiment_run"

for filename in os.listdir(data_path):
    # Construct the full file path
    base_name, extension = os.path.splitext(filename)
    if extension != '.csv':
        continue
    file_path = os.path.join(data_path, filename)

    for target_column in target_columns:
        target_folder = f"{folder_path}/{target_column}"
        tracking_folder = f"{target_folder}/Performance Tracking"
        track_file = f"{tracking_folder}/model_performance_tracking_{base_name}.json"
        save_path = f"{target_folder}/Model Weights/{base_name}"
        history_path = f"{target_folder}/Model Training History/{base_name}"

        # Create directories if they do not exist
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(tracking_folder, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(history_path, exist_ok=True)

        best_loss = float('inf')
        best_model = None
        best_model_config = None

        # Generate all combinations and create sweeps
        for (history_length, prediction_length, model_type, input_type) in itertools.product(history_lengths, prediction_lengths, model_types, input_types):
            sweep_config = create_sweep_config(target_column=target_column, history_length=history_length, prediction_length=prediction_length, model_type=model_type, input_type=input_type, save_path=save_path, data_path=file_path, history_path=history_path, probabilistic=probabilistic)
            sweep_name = f"{history_length}-{prediction_length}-{input_type}-{model_type}"
            sweep_id = wandb.sweep(sweep_config, project=f"{project_name}")
            print(f"Created sweep {sweep_id} for {sweep_name}")
            wandb.agent(sweep_id, model_training, count=count)


        if __name__ == '__main__':
            print("Setup complete. Please start the sweeps as needed.")