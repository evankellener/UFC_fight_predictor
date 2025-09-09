# UFC Fight Predictor

A machine learning application that predicts the outcome of UFC fights using fighter statistics, ELO ratings, and advanced feature engineering.

## Features

- **Fight Prediction**: Predict the winner of upcoming UFC fights
- **ELO Decay**: Accounts for fighters who haven't fought in over 365 days
- **Age Calculation**: Uses date of birth for accurate age calculations
- **Postcomp Stats**: Uses post-fight statistics from previous fights to predict upcoming fights
- **Multiple Interfaces**: Command-line, web app, and Python API
- **Model Validation**: Validates predictions on test data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd UFC_fight_predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required data files:
   - `data/tmp/final.csv` - Main fighter data
   - `data/tmp/final_data_with_dob.csv` - Date of birth data (optional)

## Usage

### Command Line Interface

#### Basic Prediction
```bash
python src/ufc_predictor_cli.py --fighter1 "Nassourdine Imavov" --fighter2 "Caio Borralho" --weight-class 10
```

#### With Custom Date
```bash
python src/ufc_predictor_cli.py --fighter1 "Jon Jones" --fighter2 "Stipe Miocic" --weight-class 12 --fight-date "2024-06-15"
```

#### List Available Fighters
```bash
python src/ufc_predictor_cli.py --list-fighters
```

#### List Weight Classes
```bash
python src/ufc_predictor_cli.py --list-weight-classes
```

#### Get Fighter Info
```bash
python src/ufc_predictor_cli.py --fighter-info "Jon Jones"
```

#### Validate Model
```bash
python src/ufc_predictor_cli.py --validate
```

### Web Application

1. Start the web server:
```bash
python src/ufc_predictor_web_app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Use the web interface to:
   - Select fighters from dropdown menus
   - Choose weight class
   - Set fight date
   - View prediction results with detailed statistics

### Python API

```python
from src.ufc_fight_predictor_app import UFCFightPredictor
from datetime import datetime, timedelta

# Initialize predictor
predictor = UFCFightPredictor()

# Make a prediction
fighter1 = "Nassourdine Imavov"
fighter2 = "Caio Borralho"
weight_class = 10  # Middleweight
fight_date = datetime.now() + timedelta(days=30)

result = predictor.predict_fight(fighter1, fighter2, weight_class, fight_date)

print(f"Predicted Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"American Odds: {result['american_odds']}")
```

## Weight Classes

| ID | Weight Class |
|----|--------------|
| 1  | Strawweight (115 lbs) |
| 2  | Women's Flyweight (125 lbs) |
| 3  | Women's Bantamweight (135 lbs) |
| 4  | Women's Featherweight (145 lbs) |
| 5  | Flyweight (125 lbs) |
| 6  | Bantamweight (135 lbs) |
| 7  | Featherweight (145 lbs) |
| 8  | Lightweight (155 lbs) |
| 9  | Welterweight (170 lbs) |
| 10 | Middleweight (185 lbs) |
| 11 | Light Heavyweight (205 lbs) |
| 12 | Heavyweight (265 lbs) |

## Model Details

### Features Used
The model uses 33 features including:
- Age and age ratio differences
- ELO ratings (overall, striking, grappling)
- Fight statistics (takedowns, strikes, accuracy)
- Physical attributes (height, weight, reach)
- Recent performance metrics

### Training Data
- **Training Period**: 2009-2023
- **Test Period**: 2024-present
- **Algorithm**: Logistic Regression with regularization
- **Preprocessing**: Standard scaling and median imputation

### Key Features

1. **Postcomp Stats Usage**: Uses post-fight statistics from previous fights to predict upcoming fights
2. **ELO Decay**: Applies 0.978 decay factor per year for fighters inactive >365 days
3. **Age Calculation**: Calculates exact age from date of birth when available
4. **Feature Mapping**: Maps precomp features to postcomp stats with fallbacks

## Example Prediction

For Nassourdine Imavov vs Caio Borralho at 185lbs:

```bash
python src/ufc_predictor_cli.py --fighter1 "Nassourdine Imavov" --fighter2 "Caio Borralho" --weight-class 10
```

This will:
1. Get each fighter's most recent postcomp stats
2. Apply ELO decay if needed (Borralho's last fight was 2024-08-24, so his ELO is decayed)
3. Calculate age ratios from DOB
4. Make prediction using the trained model
5. Generate an inference example CSV

## Validation

The model can be validated on test data to ensure the prediction process works correctly:

```bash
python src/ufc_predictor_cli.py --validate
```

This runs predictions on all test data fights using the same process as real predictions, providing accuracy metrics and sample results.

## File Structure

```
UFC_fight_predictor/
├── src/
│   ├── ufc_fight_predictor_app.py      # Main predictor class
│   ├── ufc_predictor_cli.py            # Command-line interface
│   ├── ufc_predictor_web_app.py        # Flask web application
│   └── templates/
│       └── ufc_predictor.html          # Web interface template
├── data/
│   └── tmp/
│       ├── final.csv                   # Main fighter data
│       └── final_data_with_dob.csv     # Date of birth data
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## API Endpoints (Web App)

- `GET /` - Main prediction interface
- `GET /api/fighters` - List available fighters
- `GET /api/weight-classes` - List weight classes
- `GET /api/fighter-versions/<name>` - Get fighter fight history
- `POST /api/predict` - Make fight prediction
- `GET /api/validate` - Run model validation
- `GET /api/stats` - Get model statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a machine learning model for educational and research purposes. Fight predictions should not be used for gambling or betting purposes. Always gamble responsibly.