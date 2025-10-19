# F1 Race Prediction System

A machine learning-based system for predicting Formula 1 race outcomes using historical performance data and the FastF1 API. The system analyzes driver and team performance metrics to predict race positions.

## Features

- 🏎 Real-time F1 data collection using FastF1 API
- 📊 Historical performance analysis
- 🤖 Machine learning-based predictions using ExtraTrees model
- 🎯 Race position predictions for specific events
- 📈 Driver and team performance tracking

## Architecture

The project uses a modular architecture:
- Data Collection: FastF1 API integration with caching
- Data Processing: Performance metric calculation and feature engineering
- Model Training: Scikit-learn based ML pipeline
- Prediction Interface: CLI-based prediction system

## Requirements

- Python 3.12 or higher
- FastF1 API access
- Sufficient storage for caching race data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rohith-2/F1.git
cd F1
```

2. Install dependencies using uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
uv sync
```

3. Configure the cache directory:
```python
# The cache directory is set to './.cache' by default
# You can modify this in src/utils.py and src/update_team_perf.py
fastf1.Cache.enable_cache('./.cache')
```

## Usage

### Data Collection

Update historical performance data:
```bash
python src/update_team_perf.py
```
This script:
- Fetches qualifying session data from 2018-2025
- Implements rate limiting for API calls
- Updates team and driver performance metrics

### Making Predictions

Predict race outcomes using the CLI:
```bash
python main.py --race "Monza" --year 2025
```

Required arguments:
- `--race`: Race name (must match names in `src/utils.py`)
- `--year`: Race year

### Model Training

To retrain the model:
1. Open `notebooks/training_ml.ipynb`
2. Execute all cells in sequence
3. The new model will be saved to `model/extratrees_model.joblib`

## Project Structure

```
.
├── config.yaml           # Model configuration
├── main.py              # CLI prediction interface
├── model/               # Trained model storage
├── notebooks/           # Jupyter notebooks for analysis
├── pyproject.toml       # Project dependencies
└── src/
    ├── update_team_perf.py  # Data collection
    └── utils.py             # Shared utilities
```

### Best Practices

1. **Rate Limiting**: Always maintain 5-second delays between FastF1 API calls
2. **Data Consistency**: Use standard race names from `race_int` dictionary
3. **Caching**: Enable FastF1 cache before any API operations
4. **Model Updates**: Document any changes to feature engineering in notebooks

### Known Issues

1. FastF1 API can be unstable - implement proper error handling
2. Race names must exactly match the `race_int` dictionary
3. Cache directory can grow large over time - clean periodically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the project's best practices
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
