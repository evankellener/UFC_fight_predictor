# Greedy Forward Search Algorithm for Automatic Feature Selection

This document explains the new **Greedy Forward Search Algorithm** implemented in the UFC Fight Predictor for automatic feature selection. This algorithm iteratively adds the best features to improve model performance until convergence.

## 🎯 Overview

The Greedy Forward Search Algorithm automatically selects the optimal set of features by:

1. **Starting** with a base set of features (`best_features`)
2. **Evaluating** each unused feature (`test_features`) by adding it to the current set
3. **Selecting** the feature that provides the best improvement in the chosen metric
4. **Repeating** until no significant improvement is found
5. **Converging** when accuracy and log loss no longer improve

## 🚀 Key Features

- **Automatic Convergence**: Stops when no significant improvement is found
- **Flexible Metrics**: Optimize for either `log_loss` or `accuracy`
- **Configurable Thresholds**: Customize convergence and improvement thresholds
- **Progress Tracking**: Real-time progress bars with `tqdm` showing iteration and feature evaluation progress
- **Comprehensive Logging**: Detailed progress tracking for each iteration
- **Visualization**: Built-in plotting and results visualization
- **Multiple Strategies**: Support for different search strategies (aggressive, conservative, etc.)

## 📊 Algorithm Workflow

```
1. Initialize best_features = [initial_features]
2. Initialize test_features = [all_features - best_features]
3. WHILE test_features is not empty AND improvement > threshold:
   a. For each feature in test_features:
      - Add feature to best_features
      - Train model and calculate metrics
      - Record accuracy and log_loss
   b. Select feature with best metric improvement
   c. Move selected feature from test_features to best_features
   d. Update best_metric_value
4. Return final feature set and results
```

## 📈 Progress Tracking

The algorithm includes comprehensive progress tracking with two levels of progress bars:

### **Main Iteration Progress Bar**
- Shows overall search progress
- Displays current iteration number
- Shows feature counts (best_features vs test_features)
- Displays current metric value
- Estimates remaining time

### **Feature Evaluation Progress Bar**
- Shows progress within each iteration
- Displays current feature being evaluated
- Shows accuracy and log loss for each feature
- Updates in real-time as features are tested

### **Example Progress Display**
```
🔍 Greedy Forward Search: 100%|██████████| 5/5 [02:15<00:00, 27.1s/iteration]
🔍 Evaluating features: 100%|██████████| 45/45 [01:30<00:00, 1.5s/feature]
```

## 🔧 Usage

### Basic Usage

```python
from ensemble_model_best import FightOutcomeModel

# Initialize model (data is loaded and prepared automatically)
model = FightOutcomeModel("data/final.csv")  # Provide your data file path

# Run greedy forward search
results = model.greedy_forward_search()

# Visualize results
model.visualize_greedy_search_results(results)
```

### Advanced Usage with Custom Parameters

```python
# Custom search with specific parameters
results = model.greedy_forward_search(
    initial_features=['feature1', 'feature2'],  # Start with specific features
    convergence_threshold=0.001,                 # Stop when improvement < 0.001
    max_iterations=50,                          # Maximum iterations
    metric='log_loss',                          # Optimize for log loss
    min_improvement=0.0001                     # Minimum improvement to add feature
)
```

## 📋 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_features` | list | `None` | Starting features (defaults to `importance_columns`) |
| `convergence_threshold` | float | `0.001` | Minimum improvement to continue search |
| `max_iterations` | int | `50` | Maximum number of iterations |
| `metric` | str | `'log_loss'` | Metric to optimize (`'log_loss'` or `'accuracy'`) |
| `min_improvement` | float | `0.0001` | Minimum improvement to add a feature |

## 📈 Return Values

The algorithm returns a comprehensive dictionary with:

```python
{
    'best_features': list,           # Final selected features
    'test_features': list,           # Remaining unused features
    'iteration_history': list,      # Results for each iteration
    'convergence_reason': str,       # Why the algorithm stopped
    'final_metrics': dict,           # Final accuracy and log loss
    'total_iterations': int          # Number of iterations completed
}
```

## 🎨 Visualization

The `visualize_greedy_search_results()` method creates comprehensive plots showing:

1. **Accuracy vs Iterations**: How accuracy improves over time
2. **Log Loss vs Iterations**: How log loss decreases over time
3. **Improvement vs Iterations**: Improvement magnitude over time
4. **Feature Count vs Iterations**: Number of features in each set over time

## 📊 Example Results

```
🚀 Starting Greedy Forward Search Algorithm
============================================================
📊 Initial setup:
   - Best features: 10 features
   - Test features: 45 features
   - Optimizing for: log_loss
   - Convergence threshold: 0.001

🔄 Iteration 1
----------------------------------------
   Best candidate: fighter_age_difference
   Accuracy: 0.7234
   Log Loss: 0.4567
   Improvement in log_loss: 0.0123
   ✅ Added fighter_age_difference to best features
   📈 Best features now: 11 features
   📉 Test features remaining: 44 features

🏁 Greedy Forward Search Complete
============================================================
📊 Final Results:
   - Best features: 15 features
   - Test features remaining: 40 features
   - Iterations completed: 5
   - Convergence reason: Improvement below convergence threshold (0.001)
   - Final accuracy: 0.7456
   - Final log loss: 0.4234
```

## 🔍 Search Strategies

### 1. Basic Strategy (Default)
```python
results = model.greedy_forward_search()
```
- Uses default parameters
- Optimizes for log loss
- Moderate convergence threshold

### 2. Aggressive Strategy
```python
results = model.greedy_forward_search(
    convergence_threshold=0.0005,
    max_iterations=30,
    min_improvement=0.0005
)
```
- Lower thresholds for more iterations
- Finds more features
- Takes longer to run

### 3. Conservative Strategy
```python
results = model.greedy_forward_search(
    convergence_threshold=0.01,
    max_iterations=10,
    min_improvement=0.001
)
```
- Higher thresholds for faster convergence
- Fewer features selected
- Faster execution

### 4. Accuracy-Optimized Strategy
```python
results = model.greedy_forward_search(
    metric='accuracy',
    convergence_threshold=0.0005
)
```
- Optimizes for accuracy instead of log loss
- May select different features
- Useful when accuracy is more important than probability calibration

## 🎯 Best Practices

1. **Start Small**: Begin with a small set of initial features
2. **Monitor Convergence**: Watch for early convergence to avoid overfitting
3. **Cross-Validate**: Always validate results on held-out data
4. **Compare Strategies**: Try different parameter combinations
5. **Visualize Results**: Use the built-in visualization tools
6. **Document Decisions**: Keep track of why certain features were selected

## 🚨 Important Notes

- **Computational Cost**: Each iteration evaluates all remaining features
- **Overfitting Risk**: More features don't always mean better performance
- **Metric Choice**: Choose the metric that aligns with your business goals
- **Convergence**: The algorithm may stop early if no good features remain
- **Feature Interactions**: Greedy search doesn't consider feature interactions

## 📁 Files

- **`ensemble_model_best.py`**: Main implementation with `greedy_forward_search()` method
- **`greedy_forward_search_example.py`**: Comprehensive example script
- **`GREEDY_FORWARD_SEARCH_README.md`**: This documentation

## 🏃‍♂️ Quick Start

1. **Run the example script**:
   ```bash
   python greedy_forward_search_example.py
   ```

2. **Use in your code**:
   ```python
   from ensemble_model_best import FightOutcomeModel
   
   model = FightOutcomeModel("data/final.csv")  # Data loaded automatically
   
   # Run the search
   results = model.greedy_forward_search()
   
   # Get the best features
   best_features = results['best_features']
   print(f"Selected {len(best_features)} features: {best_features}")
   ```

## 🔧 Troubleshooting

### Common Issues

1. **No features found**: Check if `main_stats_cols` contains valid features
2. **Early convergence**: Lower `convergence_threshold` or `min_improvement`
3. **Too many iterations**: Increase `convergence_threshold` or `min_improvement`
4. **Memory issues**: Reduce `max_iterations` or use fewer initial features

### Performance Tips

1. **Start with fewer features**: Use a subset of `importance_columns`
2. **Use appropriate thresholds**: Balance between thoroughness and speed
3. **Monitor progress**: Watch the iteration logs for insights
4. **Save results**: Store successful feature sets for reuse

---

**Happy Feature Selecting! 🎯**
