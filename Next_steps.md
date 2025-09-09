# Next steps

## 1. Data Quality & Pipeline
- Ensure consistency with ROI calculator
- Validate and clean all input data (handle missing values, outliers, and type mismatches)
- Standardize data formats (dates, weights, heights, etc.)
- Automate data validation checks after each pipeline step
- Add unit tests for data processing scripts

## 2. Feature Engineering & Modeling
- Add stat ELOs and other advanced features
    - UFC age (how long have they been active in the UFC)
    - Start with base ELO
    - Automate discovery of fine-tuning ELOs with LLMs
        - ELO
        - Striking ELO
            * Head (offense, defense)
            * Body
            * Clinch
            * Ground (use with grappling ELO?)
        - Grappling ELO
            * Takedown ELO
            * Submission ELO
- Feature importance analysis and selection
- Hyperparameter optimization (Optuna, MLflow)
- Cross-validation and robust model evaluation
- Add model explainability (SHAP, feature importances)


## 2.5 article

## 3. Code Quality & Structure
- Refactor code for modularity (separate data, features, models, utils)
- Add docstrings and type hints to all functions/classes
- Remove unused code and files
- Standardize naming conventions and file organization
- Add logging and error handling throughout the codebase
- Add unit and integration tests for all modules

## 4. Documentation & Reproducibility
- Update and expand README with clear usage instructions
- Add example scripts and Jupyter notebooks for common workflows
- Document all configuration files and parameters
- Add a requirements.txt or environment.yml for easy setup
- Provide a Makefile or bash scripts for common tasks (setup, train, evaluate)
- Add Sphinx or MkDocs documentation for API reference

## 5. Experiment Tracking & Results
- Use MLflow for experiment tracking and model versioning
- Save all model artifacts, metrics, and feature sets
- Maintain a changelog of major improvements and results
- Visualize model performance and feature importances

## 6. User Experience & Interface
- Add a CLI or web interface for making predictions
- Provide clear error messages and help options
- Add example input/output files for users
- Consider packaging as a Python module or Docker container

## 7. Advanced Data & Modeling
- Analyze video data (diffusion vs transformer-based approaches)
- Integrate betting odds and external data sources
- Explore ensemble and stacking models
- Research and implement new MMA-specific features

## 8. Project Management
- Set up version control best practices (branching, PRs, code review)
- Add issue templates and contribution guidelines
- Regularly update project roadmap and next steps