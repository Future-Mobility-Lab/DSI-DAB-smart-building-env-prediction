import os
import click
import wandb
import itertools
import tensorflow as tf
from model_training.model_trainer import create_sweep_config, model_training

@click.command()
@click.option('--device', type=click.Choice(['cpu', 'gpu']), default='cpu', help='Select computing device (cpu/gpu)')
@click.option('--mixed-precision', is_flag=True, help='Enable mixed precision training (GPU only)')
def main(device, mixed_precision):
    """Run experiments with specified device configuration."""
    
    # Configure device settings
    if device.lower() == 'cpu':
        print("Running on CPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')
    else:
        print("Running on GPU...")
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if mixed_precision:
                    print("Enabling mixed precision...")
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
            else:
                print("No GPU devices found. Falling back to CPU...")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        except Exception as e:
            print(f"GPU setup error: {e}")
            print("Falling back to CPU...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Force TensorFlow to be hardware-agnostic
    tf.config.set_soft_device_placement(True)

    # Your experiment configuration
    history_lengths = list(range(1, 13))  # From 1 to 12
    prediction_lengths = list(range(1, 13))  # From 1 to 12
    model_types = ['CNNModel', 'LSTMModel', 'CNNLSTMModel']
    input_types = ['univariate', 'multivariate']
    target_columns = ['t']
    folder_path = 'workspace'
    data_path = f'{folder_path}/Processed Datasets/'
    probabilistic = False
    count = 50
    project_name = "HASS_DSI_experiment_run"

    # Device info for wandb logging
    device_config = {
        'device': device,
        'mixed_precision': mixed_precision if device.lower() == 'gpu' else False,
        'tensorflow_version': tf.__version__
    }

    # Process each dataset
    for filename in os.listdir(data_path):
        base_name, extension = os.path.splitext(filename)
        if extension != '.csv':
            continue
        file_path = os.path.join(data_path, filename)

        for target_column in target_columns:
            # Setup directories
            target_folder = f"{folder_path}/{target_column}"
            tracking_folder = f"{target_folder}/Performance Tracking"
            track_file = f"{tracking_folder}/model_performance_tracking_{base_name}.json"
            save_path = f"{target_folder}/Model Weights/{base_name}"
            history_path = f"{target_folder}/Model Training History/{base_name}"

            # Create directories
            for dir_path in [target_folder, tracking_folder, save_path, history_path]:
                os.makedirs(dir_path, exist_ok=True)

            print(f"\nProcessing dataset: {base_name}")
            print(f"Target column: {target_column}")
            print(f"Device configuration: {device_config}")

            # Generate combinations and create sweeps
            combinations = list(itertools.product(
                history_lengths, 
                prediction_lengths, 
                model_types, 
                input_types
            ))
            
            total_combinations = len(combinations)
            print(f"Total experiment combinations: {total_combinations}")

            for idx, (history_length, prediction_length, model_type, input_type) in enumerate(combinations, 1):
                sweep_config = create_sweep_config(
                    target_column=target_column,
                    history_length=history_length,
                    prediction_length=prediction_length,
                    model_type=model_type,
                    input_type=input_type,
                    save_path=save_path,
                    data_path=file_path,
                    history_path=history_path,
                    probabilistic=probabilistic,
                    track_file=track_file
                )
                
                # Add device configuration to sweep config
                sweep_config['device_config'] = device_config
                
                sweep_name = f"{history_length}-{prediction_length}-{input_type}-{model_type}"
                print(f"\nCreating sweep {idx}/{total_combinations}: {sweep_name}")
                
                sweep_id = wandb.sweep(sweep_config, project=f"{project_name}")
                print(f"Created sweep {sweep_id} for {sweep_name}")
                wandb.agent(sweep_id, model_training, count=count)

if __name__ == '__main__':
    main()