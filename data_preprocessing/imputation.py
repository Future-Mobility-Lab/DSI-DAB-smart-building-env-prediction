import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def calculate_correlation(main_data, other_data):
    # Make sure to only correlate like-columns
    common_columns = main_data.columns.intersection(other_data.columns)
    return main_data[common_columns].corrwith(other_data[common_columns], axis=0).mean()

# Function to calculate the correlation with other sensors
def find_correlated_sensors(main_data, folder_path, current_file_name):
    correlations = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file != current_file_name: # Exclude the current file
            file_path = os.path.join(folder_path, file)
            other_data = pd.read_csv(file_path, parse_dates=['Unnamed: 0.1'])
            other_data.set_index('Unnamed: 0.1', inplace=True)
            correlation = calculate_correlation(main_data, other_data)
            correlations[file] = correlation
    return correlations

def knn_impute_sensor(data, correlated_data, n_neighbors=12):
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(correlated_data)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_data_scaled = imputer.fit_transform(combined_data_scaled)
    imputed_data = scaler.inverse_transform(imputed_data_scaled)
    imputed_df = pd.DataFrame(imputed_data, columns=correlated_data.columns, index=correlated_data.index)
    return imputed_df[data.columns] # Return only the original columns

def custom_imputer(data, folder_path, current_file_name):
    # Mean imputation for one-hour gaps
    for column in data.columns:
        missing_indices = data.index[data[column].isna()]
        for missing_index in missing_indices:
            prev_index = missing_index - pd.Timedelta(hours=1)
            next_index = missing_index + pd.Timedelta(hours=1)
            if prev_index in data.index and next_index in data.index and not data[column].isna().loc[prev_index] and not data[column].isna().loc[next_index]:
                data.at[missing_index, column] = (data.at[prev_index, column] + data.at[next_index, column]) / 2

    # Find the most correlated sensors
    correlated_sensors = find_correlated_sensors(data, folder_path, current_file_name)

    # Sort sensors by correlation and take the most correlated one
    top_sensor_file = sorted(correlated_sensors, key=correlated_sensors.get, reverse=True)[0]

    # Print the most correlated dataset
    print(f"The most correlated dataset for {current_file_name} is {top_sensor_file} with a correlation of {correlated_sensors[top_sensor_file]}.")

    file_path = os.path.join(folder_path, top_sensor_file)
    top_sensor_data = pd.read_csv(file_path, parse_dates=['Unnamed: 0.1'])
    top_sensor_data.set_index('Unnamed: 0.1', inplace=True)

    # Combine data with the top correlated sensor
    combined_data = pd.concat([data, top_sensor_data], axis=1)

    # Impute missing values using k-NN
    imputed_data = knn_impute_sensor(data, combined_data)

    return imputed_data[data.columns]

def process_all_datasets(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, parse_dates=['Unnamed: 0.1'])
            data.set_index('Unnamed: 0.1', inplace=True)

            try:
                # Apply the custom imputer
                data_imputed = custom_imputer(data, folder_path, file)

                # Save the imputed data back to the folder with 'imputed_' prefix
                output_path = os.path.join(folder_path, 'imputed_' + file)
                data_imputed.to_csv(output_path)
                print(f"Dataset {file} processed successfully!")
            except ValueError as e:
                # Check if the error message matches the specific case
                if "operands could not be broadcast together with shapes" in str(e):
                    print(f"Dataset {file} skipped due to error: {e}")
                    continue
                else:
                    raise e

    print("All datasets processed!")

processed_folder_path = '/content/drive/MyDrive/UTS/Hass-DSI/Device Data/Device Data (Updated)/processed/'
process_all_datasets(processed_folder_path)