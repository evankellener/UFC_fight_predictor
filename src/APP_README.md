# 🥊 Enhanced UFC Fight Predictor App

A Flask-based web application that provides advanced UFC fight predictions using machine learning models with fighter version selection and bidirectional averaging.

## ✨ Features

- **Advanced Fight Predictions**: Uses enhanced ELO ratings and machine learning models
- **Fighter Version Selection**: Choose which post-fight stats to use for predictions
- **Bidirectional Averaging**: Combines predictions from both directions for better accuracy
- **Web Interface**: Beautiful, responsive UI for easy predictions
- **RESTful API**: Full API access for integration with other applications
- **Real-time Statistics**: View model performance and system statistics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository and navigate to the src directory:**
   ```bash
   cd src/
   ```

2. **Activate the virtual environment:**
   ```bash
   source ../ufc_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

4. **Start the application:**
   ```bash
   python app.py
   ```

5. **Open your browser and navigate to:**
   ```
   http://localhost:8080
   ```

## 🌐 Web Interface

The web interface provides:
- **Fighter Selection**: Autocomplete search for fighter names
- **Version Selection**: Choose specific post-fight stat versions
- **Prediction Results**: Detailed winner prediction with confidence and odds
- **Fighter Comparison**: Side-by-side stats comparison
- **System Statistics**: Model performance and data overview

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `GET` | `/api` | API documentation |
| `GET` | `/fighters` | List all fighters |
| `GET` | `/fighter/<name>` | Fighter details |
| `GET` | `/fighter/<name>/versions` | Fighter versions |
| `GET` | `/stats` | Model statistics |
| `POST` | `/predict` | Make prediction (recent versions) |
| `POST` | `/predict_with_versions` | Make prediction (specific versions) |

### Example API Usage

#### Make a Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"fighter1": "Alex Pereira", "fighter2": "Magomed Ankalaev"}'
```

#### Get Fighter Versions
```bash
curl http://localhost:8080/fighter/Alex%20Pereira/versions
```

#### Health Check
```bash
curl http://localhost:8080/health
```

## 🧠 Prediction Model

The app uses an enhanced UFC predictor that includes:

- **Logistic Regression Model**: Trained on 10,920 fights with 36 features
- **Enhanced ELO Ratings**: Post-fight ELO, striking ELO, and grappling ELO
- **Bidirectional Averaging**: Combines predictions from both fighter perspectives
- **Feature Engineering**: Age ratios, accuracy percentages, takedown stats, etc.
- **Consistent Normalization**: Proper scaling and imputation for training/testing

### Model Performance
- **Training Accuracy**: 61.4%
- **Test Accuracy**: 66.7%
- **Features Used**: 36
- **Data Range**: 2009-2025
- **Total Fighters**: 1,620

## 🏗️ Architecture

```
app.py (Flask Application)
├── EnhancedUFCPredictor
│   ├── Data Loading & Preprocessing
│   ├── Feature Engineering
│   ├── Model Training
│   └── Prediction Engine
├── Web Interface (HTML/CSS/JS)
├── REST API Endpoints
└── Error Handling & Validation
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Server port (default: 8080)
- `HOST`: Server host (default: 0.0.0.0)

### Data Paths
- **Fighter Data**: `../data/tmp/final.csv`
- **Model Files**: Automatically generated and cached
- **Logs**: `app.log` in the src directory

## 📊 Data Sources

The app uses comprehensive UFC fight data including:
- Fight outcomes and statistics
- Fighter demographics (age, height, weight, reach)
- Historical performance metrics
- ELO rating calculations
- Post-fight statistics

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -ti:8080 | xargs kill -9
   ```

2. **Predictor Not Available**
   - Check if data file exists at `../data/tmp/final.csv`
   - Verify virtual environment is activated
   - Check app.log for error messages

3. **Model Training Issues**
   - Ensure sufficient memory for large datasets
   - Check feature column availability
   - Verify data quality and completeness

### Logs
- Application logs: `app.log`
- Check for initialization errors and startup issues

## 🔒 Security Notes

- **Development Server**: This is a development server, not for production use
- **CORS Enabled**: Cross-origin requests are allowed for development
- **Debug Mode**: Enabled for development (disable in production)

## 📈 Performance

- **Startup Time**: ~30-60 seconds (includes model training)
- **Prediction Time**: <1 second per prediction
- **Memory Usage**: ~2-4GB during operation
- **Concurrent Users**: Supports multiple simultaneous requests

## 🤝 Contributing

To improve the app:
1. Enhance the prediction model in `enhanced_predictor.py`
2. Improve the web interface in `templates/enhanced_index.html`
3. Add new API endpoints in `app.py`
4. Update feature engineering and data preprocessing

## 📝 License

This project is part of the UFC Fight Predictor system.

## 🆘 Support

For issues or questions:
1. Check the logs in `app.log`
2. Verify data file availability
3. Ensure all dependencies are installed
4. Check virtual environment activation

---

**Happy Predicting! 🥊🏆**
