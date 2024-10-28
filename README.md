# DSI-DAB-smart-building-env-prediction

This repository contains the codebase for a collaborative research project between the Data Science Institute (DSI) and the Faculty of Design, Architecture and Building (DAB). The project focuses on optimising indoor environmental prediction in smart buildings using deep learning models.

<!-- Add your banner image here -->
<div align="center">
  <img src="assets/uts-logo.png" alt="Project Banner" width="100%">
</div>

## Prerequisites

- Python 3.10
- Git

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/DSI-DAB-smart-building-env-prediction.git
cd DSI-DAB-smart-building-env-prediction
```

2. Create and activate a virtual environment:
```bash
python3.10 -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
workspace/
├── Processed Datasets/     # Contains preprocessed CSV files
├── t/                     # Target variable specific folders
│   ├── Model Weights/     # Trained model weights
│   ├── Model Training History/  # Training history files
│   └── Performance Tracking/ # Model tracking
└── Evaluation/           # Model evaluation results
    └── Figures/          # Generated visualizations including Shapley plots
```

## Running the Project

### Training Models

The project uses Weights & Biases (wandb) for experiment tracking. To train models, run:

```bash
python experiment-run.py --device [cpu/gpu] --mixed-precision
```

This script performs a comprehensive grid search across multiple configurations:
- History lengths: 1 to 12 timesteps
- Prediction horizons: 1 to 12 timesteps
- Model architectures: CNN, LSTM, and CNN-LSTM hybrid
- Input types: univariate and multivariate

Options:
- `--device`: Choose between 'cpu' or 'gpu' (default: 'cpu')
- `--mixed-precision`: Enable mixed precision training for GPU (optional flag)

### Evaluating Models

To evaluate trained models and generate performance metrics, simply run:

```bash
python experiment-test.py
```

This script will:
1. Process all models in the Model Weights directory
2. Calculate performance metrics (RMSE, MAPE, SMAPE)
3. Generate visualization plots
4. Save results to a CSV file in the Evaluation directory

### Shapley Analysis

To perform feature importance analysis using Shapley values, run:

```bash
python shapley-run.py
```

Note: The Shapley analysis script has been specifically configured and verified for:
- History length: 12 timesteps (fixed)
- Prediction horizon: 1 timestep (fixed)
- Input type: multivariate only
- Model architectures: CNN, LSTM, and CNN-LSTM hybrid
- These parameters are currently hard-coded in the script

This script generates:
- Temporal Shapley value plots showing feature contributions across timesteps
- Feature importance bar plots based on mean absolute SHAP values
- Detailed statistics of feature impacts saved in CSV format
- Visualizations saved in the `workspace/Evaluation/Figures/` directory

## Output

The evaluation process generates several outputs:
- Performance metrics in CSV format
- Prediction plots comparing actual vs predicted values
- Training history plots
- Shapley value analysis plots and statistics (for multivariate models)

## License

[Your license information here]

## Contributors

- Roupen Minassian (Machine Learning Researcher, Data Science Institute)
- Adriana-Simona Mihăiţă (Associate Professor, Data Science Institute)
- Arezoo Shirazi (Senior Lecturer + Course Director, Faculty of Design, Architecture and Building)

## Contact

For questions or collaboration opportunities, please contact Roupen Minassian at firstname.lastname@uts.edu.au