# DSI-DAB-smart-building-env-prediction

This repository contains the codebase for a collaborative research project between the Data Science Institute (DSI) and the Faculty of Design, Architecture and Building. The project focuses on optimising indoor environmental prediction in smart buildings using deep learning models.

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
│   ├── Model Training History/  # Training
│   └── Performance Tracking/ # Model tracking
history files
└── Evaluation/           # Model evaluation results
```

## Running the Project

### Training Models

The project uses Weights & Biases (wandb) for experiment tracking. To train models, run:

```bash
python run_experiments.py --device [cpu/gpu] --mixed-precision
```

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

## Output

The evaluation process generates several outputs:
- Performance metrics in CSV format
- Prediction plots comparing actual vs predicted values
- Training history plots
- Peak analysis plots (for non-probabilistic models)
- Shapley value analysis (for multivariate models)

## License

[Your license information here]

## Contributors

- [List of contributors]

## Contact

For questions or collaboration opportunities, please contact [contact information].