# UFC Fight Predictor

A machine learning system for predicting UFC fight outcomes using fighter statistics and an enhanced ELO rating system.

## Features

- Data scraping pipeline for UFC fighter and fight statistics 
- SQLite database for efficient data storage and processing
- Advanced feature engineering including fighter performance metrics
- Enhanced ELO rating system for fighter comparison
- Deep Neural Network (DNN) model with hyperparameter optimization
- Prediction interface for upcoming UFC fights

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/UFC_fight_predictor.git
cd UFC_fight_predictor
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage

The main workflow is implemented in the Jupyter notebook:

```bash
jupyter lab notebooks/01_Fight_Predictor_Pipeline.ipynb
```

This notebook contains the complete pipeline:
1. Data scraping from UFC statistics websites
2. Data processing and feature engineering
3. Model training with hyperparameter optimization
4. Fight prediction

## Core Components

### Data Pipeline

- **Scraping**: Extracts fighter and fight data from official UFC statistics
- **Processing**: Cleans and transforms data into features suitable for machine learning
- **SQL Storage**: Maintains structured data in SQLite for efficient retrieval

### Enhanced ELO Rating System

The ELO system tracks fighter performance with improvements:
- Reduced decay rate (2 points per month) with maximum decay cap
- 15% bonus for title fights
- 10% adjustment for upsets
- Minimum ELO floor of 1200 points

### Predictive Model

- TensorFlow/Keras DNN model
- MLflow for experiment tracking and model versioning
- Hyperparameter optimization using Optuna
- Evaluation metrics including accuracy, ROC curves, and feature importance

## Recent Improvements

- Fixed ELO calculation issues that caused champion fighters to have unexpectedly low ratings
- Improved path handling for more reliable operation
- Enhanced hyperparameter tuning
- Fixed data type handling for more stable model training

## Prediction Example

```python
from model import DNNFightPredictor

predictor = DNNFightPredictor(file_path='data/tmp/final.csv')
predictor.predict_fight_winner('Fighter One', 'Fighter Two')
```

## Future Enhancements

To further increase prediction accuracy:
1. Ensemble modeling - combining multiple models
2. Advanced feature engineering
3. Further hyperparameter tuning
4. Cross-validation techniques
