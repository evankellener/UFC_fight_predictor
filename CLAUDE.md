# UFC Fight Predictor Project Guidelines

## Commands
- **Environment Setup**: `python -m venv .venv && source .venv/bin/activate`
- **Install Dependencies**: `pip install -r requirements.txt`
- **Run Jupyter Notebook**: `jupyter lab notebooks/01_Fight_Predictor_Pipeline.ipynb`
- **Data Scraping**: Complete pipeline in Jupyter notebook runs all scrapers sequentially
- **Model Training**: Use `predictor.hyperparameter_tuning()` in notebook for optimization
- **Test Model**: `predictor.predict_fight_winner(fighter1, fighter2)` for predictions

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then project modules
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Type Hints**: Not consistently used, but encouraged for new code
- **Data Structure**: Data is processed through SQLite into CSV files for model input
- **Error Handling**: Use try/except blocks for web scraping and file operations
- **Documentation**: Docstrings for classes, inline comments for complex logic
- **Model Structure**: TensorFlow/Keras for DNN models with MLflow for tracking

## Architecture
- Scraping → SQL Processing → Feature Engineering → Model Training → Evaluation
- Uses MLflow for experiment tracking and model versioning
- ELO rating system for fighter comparison implemented in elo_class.py


  To further increase accuracy, you could:

  1. Consider ensemble modeling - train multiple models and combine their
  predictions
  2. Experiment with feature engineering to create more predictive variables
  3. Fine-tune the hyperparameters (learning rate, layer sizes, dropout
  rates)
  4. Use cross-validation to further improve reliability

  The enhanced ELO system is now in place and ready to be used for accurate
  UFC fight predictions.

## Recent Fixes
- Fixed "Invalid dtype: object" error in the model by explicitly converting target column to numeric
- Improved path handling in code by using absolute paths
- Resolved MLflow tracking paths for more reliable training
- Enhanced hyperparameter tuning by updating Optuna parameter suggestions
- Fixed ELO calculation issues that caused champion fighters to have unexpectedly low ratings
  - Reduced decay rate from 5 to 2 points per month with a maximum decay cap
  - Added modest 15% bonus for title fights (balanced for model accuracy)
  - Added minimal upset adjustment (10%) to avoid overfitting
  - Increased minimum ELO floor from 1000 to 1200 points
  - Maintained model accuracy while improving champion ratings