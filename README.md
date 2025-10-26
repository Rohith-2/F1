# F1 Race Prediction System

An advanced machine learning system for predicting Formula 1 qualifying outcomes using FastF1 API data. The system leverages historical performance metrics, real-time session data, and ensemble learning to predict race positions and lap times.

## 🚀 Features

- 🏎️ Real-time F1 data collection via FastF1 API
- 📊 Comprehensive historical performance analysis (2018-2025)
- 🤖 Machine learning predictions using Numerous Ensemble Regressor
- 🎯 Race position and lap time predictions
- 📈 Driver and team performance tracking
- 🔄 Automated data pipeline with caching
- 📱 CLI interface for easy predictions

## 🏗️ Architecture

The project follows a modular architecture:
- **Data Collection Layer**: FastF1 API integration with smart caching
- **Processing Pipeline**: Advanced feature engineering and metric calculation
- **ML Core**: Scikit-learn based predictive modeling
- **Interface Layer**: Command-line interface for predictions

## ⚙️ Requirements

- Python ≥ 3.12
- 2GB+ free storage for race data cache
- macOS/Linux/Windows compatible
- Internet connection for FastF1 API access

## 📦 Dependencies

Core dependencies:
- `fastf1 ≥ 3.6.1` - F1 data access
- `scikit-learn ≥ 1.7.2` - Machine learning
- `pandas`, `numpy` - Data processing
- `lightgbm ≥ 4.6.0`, `xgboost ≥ 3.1.0` - Additional ML models
- `plotly ≥ 6.3.1` - Visualization
- `tqdm ≥ 4.67.1` - Progress tracking

Development dependencies:
- `jupyter ≥ 1.1.1` - Notebook interface
- `seaborn ≥ 0.13.2`, `matplotlib ≥ 3.10.6` - Data visualization

## 🛠️ Installation

1. **Environment Setup**:
   ```bash
   # Clone repository
   git clone https://github.com/Rohith-2/F1.git
   cd F1
   # Install UV package manager
   pip install uv
   
   # Create and activate virtual environment
    uv init
   
   # Install dependencies
   uv sync
   ```

2. **Cache Configuration**:
   ```bash
   # Create cache directory
   mkdir -p .cache
   
   # Configure environment variables (optional)
   export F1_CACHE_DIR="./.cache"  # Unix/macOS
   # or
   set F1_CACHE_DIR=".\.cache"     # Windows
   ```

## 📊 Usage

### Data Collection & Updates

```bash
# Update historical performance data
python src/update_team_perf.py
```

### Making Predictions

```bash
# Predict using race name
python main.py --race "Monaco" --year 2025

# Predict using race number
python main.py --race 7 --year 2025

# Export predictions to CSV
python main.py --race "Silverstone" --year 2025
```

### Model Training

1. **Prepare Training Data**:
   ```python
   jupyter notebook notebooks/prep_training_data.ipynb
   # Execute all cells to generate training datasets
   ```

2. **Train Model**:
   ```python
   jupyter notebook notebooks/training_ml.ipynb
   # Execute cells sequentially
   # Model will be saved to model/model.joblib
   ```

3. **Validate Model**:
   ```python
   jupyter notebook notebooks/predict.ipynb
   # Run prediction validation notebook
   ```

## 📁 Project Structure

```
F1/
├── .cache/              # FastF1 data cache
├── model/
│   ├── model.joblib     # Trained model
│   ├── model_leaderboard.pkl
│   └── team_driver_performance.csv
├── notebooks/
│   ├── init.ipynb       # Environment setup
│   ├── predict.ipynb    # Prediction examples
│   ├── prep_training_data.ipynb
│   ├── team_data.ipynb  # Team analysis
│   └── training_ml.ipynb # Model training
├── src/
│   ├── train.py        # Training pipeline
│   ├── update_team_perf.py
│   └── utils.py        # Shared utilities
├── main.py             # CLI interface
├── pyproject.toml      # Dependencies
└── README.md
```

## 🔍 Best Practices

1. **API Usage**:
   - Implement 5-second delays between FastF1 API calls
   - Use error handling for API failures
   - Cache data to minimize API requests

2. **Data Management**:
   - Use standardized race names from `race_int` dictionary
   - Regular cache cleanup (keep last 2 seasons)
   - Backup model files before retraining

3. **Model Development**:
   - Document feature engineering changes
   - Version control model files
   - Regular performance validation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 📊 Citations

If you use this project in your research, please cite:

```bibtex
@software{F1_Predictor,
  author = {Rohith},
  title = {F1 Race Prediction System},
  year = {2025},
  url = {https://github.com/Rohith-2/F1}
}
```
