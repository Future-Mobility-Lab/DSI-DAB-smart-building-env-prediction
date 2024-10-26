import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from model_training.data_preparer import DataPreparer

folder_path = 'workspace'
target_column = "t"
data_path = f'{folder_path}/Processed Datasets/'
model_dir = f'{folder_path}/Evaluation/{target_column}/'
history_length = 12
prediction_length = 1
input_type = 'multivariate'
model_type = ['CNNModel', 'LSTMModel', 'CNNLSTMModel']
probabilistic = False
results_folder = f'{folder_path}/Evaluation/{target_column}/'
history_folder = f'{folder_path}/{target_column}/Model Training History/'

# List to store all experiment results
all_experiments = []

# Evaluate all models
for filename in os.listdir(model_dir):

  experiment_details = {'Shapley': []}

  if 'result' in filename.lower():  # Skip files with 'result' in their names
    continue

  data = pd.read_csv(data_path+filename+'.csv')
  shapley_filename = model_dir+filename+'/Shapley Values/'+f"{history_length}-{prediction_length}-{input_type}-{model_type}_shap_values.npy"
  data_preparer = DataPreparer(data=data, target_column=target_column, history_length=history_length, prediction_length=prediction_length, input_type=input_type, datetime_column='Time')

  shapley_values_list = np.load(shapley_filename, allow_pickle=True)  # Assuming this is a list of dictionaries

  for shapley_values in shapley_values_list:
      experiment_details['Shapley'].append(shapley_values)

  experiment_details['columns'] = data_preparer.get_column_names()
  experiment_details['start_date'] = data_preparer.get_split_start_dates()

  all_experiments.append(experiment_details)

# Prepare to accumulate all reshaped data from experiments
all_reshaped_shap_values = []
feature_sets = []
timesteps_per_sample = 12
quarters = {0: '00:00-05:59', 1: '06:00-11:59', 2: '12:00-17:59', 3: '18:00-23:59'}

for experiment in all_experiments:
    feature_names = experiment['columns']
    feature_sets.append(set(feature_names))

    shap_values_final = [result['shap_values'] for result in experiment['Shapley']]
    reshaped_shap_values = [values.squeeze() for values in shap_values_final]
    flat_shap_values = np.vstack(reshaped_shap_values)

    start_hour = experiment['start_date']['test_start'].hour
    num_samples = len(reshaped_shap_values)
    predicted_hours = np.tile(np.arange(24), num_samples // 24 + 1)[:num_samples]
    predicted_hours = (predicted_hours + start_hour) % 24

    # Create a DataFrame for this experiment
    df = pd.DataFrame(data=flat_shap_values, columns=feature_names)
    df['Experiment'] = len(all_reshaped_shap_values)  # Assign an experiment index
    df['Timestep'] = np.tile(np.arange(timesteps_per_sample), len(df) // timesteps_per_sample)
    df['Predicted_Hour'] = np.repeat(predicted_hours, timesteps_per_sample)
    all_reshaped_shap_values.append(df)

# Identify common features across all experiments
common_features = set.intersection(*feature_sets)
all_features = set.union(*feature_sets)

# Align data by ensuring all DataFrames have the same columns
aligned_dfs = []
for df in all_reshaped_shap_values:
    for feature in all_features:
        if feature not in df:
            df[feature] = np.nan  # or df[feature] = 0 if zero is more appropriate
    aligned_dfs.append(df)

# Concatenate all aligned DataFrames
df_shap_all = pd.concat(aligned_dfs, ignore_index=True)

# Calculate mean and variance for each feature per timestep
stats_per_timestep = df_shap_all.groupby('Timestep').agg({**{feature: ['mean', 'var'] for feature in all_features}})

# Convert standard deviation to 95% confidence interval (mean ± 1.96*std/sqrt(n))
for feature in all_features:
    n = stats_per_timestep[(feature, 'mean')].count()
    ci_multiplier = 1.96 / np.sqrt(n)
    stats_per_timestep[(feature, 'CI_lower')] = stats_per_timestep[(feature, 'mean')] - stats_per_timestep[(feature, 'var')].apply(np.sqrt) * ci_multiplier
    stats_per_timestep[(feature, 'CI_upper')] = stats_per_timestep[(feature, 'mean')] + stats_per_timestep[(feature, 'var')].apply(np.sqrt) * ci_multiplier

# Custom names for features if you want to change them
custom_feature_names = {
    't': 'Temperature',
    'People_Count': 'People Count',
    'h': 'Humidity',
    'voc': 'VOC',
    'p': 'Pressure',
    'HVAC_Status': 'HVAC Status',
    'Booked': 'Booking Status',
    'co2': 'CO2',
    'pm10': 'PM10',
    'pm25': 'PM2.5'
}

filter_items = set(custom_feature_names.keys())  # Update filter_items to ensure it matches the custom names keys if any

filtered_features = [feature for feature in all_features if feature in filter_items]
num_features = len(filtered_features)
nrows = num_features  # Now the number of rows equals the number of features
fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 2.5 * nrows))  # Single column with one plot per feature

timestep_labels = {i: f'T-{12-i}' for i in range(12)}

for plot_idx, feature in enumerate(filtered_features):
    ax = axes[plot_idx]
    means = stats_per_timestep[(feature, 'mean')]
    ci_lower = stats_per_timestep[(feature, 'CI_lower')]
    ci_upper = stats_per_timestep[(feature, 'CI_upper')]
    ax.fill_between(means.index, ci_lower, ci_upper, alpha=0.3, color='grey', zorder=1)
    ax.plot(means.index, means, color='black', linestyle='dashed', zorder=2)

    ax.grid(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Remove y-axis label since you don't want it displayed
    ax.set_ylabel('')

    # Explicitly setting the x-axis limits
    ax.set_xlim(min(means.index), max(means.index))

    if plot_idx == nrows - 1:
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels([timestep_labels[i] for i in range(12)], fontsize=16)
    else:
        ax.set_xticks([])

    # Move the title to the top left inside the subplot
    ax.text(0.02, 0.04, custom_feature_names.get(feature, feature), fontsize=18, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')

    # Add alphabetical label at the bottom left
    alphabetical_label = string.ascii_lowercase[plot_idx]
    ax.text(0.02, 0.96, alphabetical_label, fontsize=18, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')

# Minimize padding and adjust spacing
plt.subplots_adjust(hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)  # Adjust subplot parameters to minimize padding
plt.tight_layout()

plt.savefig(f'{folder_path}/Evaluation/Figures/shapley_plot_continuous.pdf', format='pdf', bbox_inches='tight')

# Save the statistics to a CSV file for reporting
stats_summary = stats_per_timestep.stack(level=0).reset_index()
stats_summary.columns = ['Timestep', 'Feature', 'Mean', 'Variance', 'CI_lower', 'CI_upper']
#stats_summary.to_csv("/content/statistics_summary.csv", index=False)

feature_list = ['h','t','People_Count','voc','Booked','p','HVAC_Status','co2'] #list(all_features)

# Calculate mean absolute Shapley values for each feature
mean_shap_abs = df_shap_all[feature_list].apply(np.abs).mean().sort_values(ascending=True)

# Plotting the feature importances
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size as needed for vertical orientation
y_positions = np.arange(len(mean_shap_abs))  # y positions for each bar
ax.barh(y_positions, mean_shap_abs.values, height=0.8, color='skyblue')  # Control height for bar width

ax.set_yticks(y_positions)
ax.set_yticklabels([custom_feature_names.get(idx, idx) for idx in mean_shap_abs.index])

ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
ax.set_yticklabels([custom_feature_names.get(idx, idx) for idx in mean_shap_abs.index])  # Map index to custom names

# Set the x-axis major locator
ax.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()

plt.savefig(f'{folder_path}/Evaluation/Figures/feature_importance_shap.pdf', format='pdf', bbox_inches='tight')