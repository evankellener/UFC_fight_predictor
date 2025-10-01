# Advanced Feature Selection Methods for UFC Fight Predictor

This document describes the advanced feature selection methods that have been added to the `ensemble_model_best.py` file. These methods provide sophisticated approaches to identifying the most predictive features for UFC fight outcomes.

## Overview

The advanced feature selection methods implement multiple state-of-the-art techniques for feature selection, specifically adapted for the UFC fight prediction domain. These methods help identify the most important features while reducing overfitting and improving model performance.

## Available Methods

### 1. Recursive Feature Elimination (RFE)
- **Purpose**: Iteratively removes the least important features
- **How it works**: Starts with all features and removes the least important ones step by step
- **Advantages**: Simple, effective, works well with linear models
- **Best for**: When you want a straightforward feature reduction approach

### 2. RFE with Cross-Validation (RFECV)
- **Purpose**: Automatically determines the optimal number of features
- **How it works**: Uses cross-validation to find the best feature subset size
- **Advantages**: Automatically selects optimal feature count, more robust than basic RFE
- **Best for**: When you're unsure about the optimal number of features

### 3. SelectFromModel with Random Forest
- **Purpose**: Uses tree-based feature importance for selection
- **How it works**: Trains a Random Forest and selects features based on importance scores
- **Advantages**: Captures non-linear relationships, handles feature interactions
- **Best for**: When you suspect non-linear feature relationships

### 4. Permutation Importance
- **Purpose**: Measures feature importance by shuffling values
- **How it works**: Randomly permutes feature values and measures performance drop
- **Advantages**: Model-agnostic, provides unbiased importance estimates
- **Best for**: When you want unbiased feature importance

### 5. PCA-based Feature Selection
- **Purpose**: Reduces dimensionality using principal components
- **How it works**: Transforms features into orthogonal components
- **Advantages**: Handles multicollinearity, reduces noise
- **Best for**: When features are highly correlated

### 6. Ensemble Feature Selection
- **Purpose**: Combines multiple methods using voting
- **How it works**: Aggregates feature selections from multiple methods
- **Advantages**: More robust, reduces bias from single methods
- **Best for**: When you want the most reliable feature selection

## Usage

### Basic Usage

```python
from ensemble_model_best import FightOutcomeModel

# Initialize model with your data
model = FightOutcomeModel("path/to/your/data.csv")

# Run advanced feature selection
results = model.advanced_feature_selection_methods(
    n_features_to_select=20,  # Number of features to select
    cv_folds=5               # Cross-validation folds
)

# Run ensemble feature selection
ensemble_results = model.advanced_ensemble_feature_selection(
    n_features_to_select=20,
    cv_folds=5
)
```

### Advanced Usage

```python
# Get detailed results
results = model.advanced_feature_selection_methods(n_features_to_select=30)

# Access specific method results
rfe_features = results['RFE']['features']
rfe_score = results['RFE']['mean_score']

# Get ensemble results
ensemble_features = ensemble_results['ensemble_features']
feature_votes = ensemble_results['feature_votes']
```

## Method Comparison

| Method | Speed | Robustness | Interpretability | Best Use Case |
|--------|-------|------------|------------------|---------------|
| RFE | Fast | Medium | High | Quick feature reduction |
| RFECV | Medium | High | High | Optimal feature count |
| SelectFromModel | Medium | High | Medium | Non-linear relationships |
| Permutation Importance | Slow | Very High | High | Unbiased importance |
| PCA | Fast | Medium | Low | Dimensionality reduction |
| Ensemble | Medium | Very High | High | Most reliable selection |

## Performance Metrics

All methods are evaluated using:
- **Log Loss**: Primary metric for probabilistic predictions
- **Cross-Validation**: 5-fold stratified cross-validation
- **Time**: Execution time for performance comparison
- **Feature Count**: Number of selected features

## Output Format

### Individual Methods
```python
{
    'RFE': {
        'scores': array([...]),           # CV scores
        'mean_score': 0.1234,            # Mean log loss
        'std_score': 0.0056,             # Std deviation
        'time': 12.34,                   # Execution time
        'features': ['feature1', ...],   # Selected features
        'n_features': 20                 # Number of features
    },
    # ... other methods
}
```

### Ensemble Results
```python
{
    'ensemble_features': ['feature1', ...],  # Final selected features
    'feature_votes': {'feature1': 3, ...},   # Vote counts
    'scores': array([...]),                  # CV scores
    'mean_score': 0.1234,                   # Mean log loss
    'std_score': 0.0056,                    # Std deviation
    'improvement': 0.0012                   # Improvement over best individual
}
```

## Best Practices

### 1. Feature Count Selection
- Start with 15-25 features for initial analysis
- Use RFECV to determine optimal count
- Consider computational constraints

### 2. Cross-Validation
- Use 5-fold CV for good balance of speed and robustness
- Ensure stratified CV for imbalanced datasets
- Use time series CV for temporal data

### 3. Method Selection
- Use ensemble method for most reliable results
- Combine multiple methods for comprehensive analysis
- Consider computational budget when choosing methods

### 4. Validation
- Always validate on held-out test set
- Compare with baseline (all features)
- Check for overfitting with learning curves

## Example Workflow

```python
# 1. Initialize model
model = FightOutcomeModel("data.csv")

# 2. Run individual methods
individual_results = model.advanced_feature_selection_methods(n_features_to_select=20)

# 3. Run ensemble method
ensemble_results = model.advanced_ensemble_feature_selection(n_features_to_select=20)

# 4. Compare results
best_individual = min(individual_results.keys(), 
                     key=lambda k: individual_results[k]['mean_score'])
print(f"Best individual: {best_individual}")
print(f"Ensemble improvement: {ensemble_results['improvement']:.4f}")

# 5. Use selected features
selected_features = ensemble_results['ensemble_features']
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `n_features_to_select` or use fewer CV folds
2. **Slow Performance**: Use faster methods (RFE, PCA) or reduce CV folds
3. **Poor Results**: Try different feature counts or combine methods
4. **Convergence Issues**: Check data quality and feature scaling

### Performance Tips

1. **Preprocessing**: Ensure proper data cleaning and scaling
2. **Feature Engineering**: Create meaningful features before selection
3. **Domain Knowledge**: Use domain expertise to guide selection
4. **Iterative Approach**: Start with fewer features and gradually increase

## Integration with Existing Methods

The advanced feature selection methods integrate seamlessly with existing functionality:

- **ROI Optimization**: Use selected features with `roi_optimized_features()`
- **Model Training**: Use with `tune_logistic_regression()`
- **Feature Analysis**: Combine with `find_best_feature_to_add()`

## Future Enhancements

Potential improvements for future versions:

1. **Parallel Processing**: Implement parallel feature selection
2. **Adaptive Selection**: Dynamic feature count based on performance
3. **Feature Interaction**: Detect and select feature interactions
4. **Online Learning**: Incremental feature selection for streaming data
5. **Visualization**: Enhanced plotting and visualization tools

## References

- Scikit-learn Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html
- Recursive Feature Elimination: Guyon et al. (2002)
- Permutation Importance: Breiman (2001)
- PCA: Jolliffe (2002)

## Support

For questions or issues with the advanced feature selection methods:

1. Check the example script: `advanced_feature_selection_example.py`
2. Review the method documentation in `ensemble_model_best.py`
3. Test with smaller datasets first
4. Ensure proper data preprocessing

---

*This documentation covers the advanced feature selection methods added to the UFC Fight Predictor. For general usage, see the main README.md file.*
