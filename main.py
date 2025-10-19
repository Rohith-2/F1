import yaml
import joblib
import logging
import argparse

import pandas as pd
from src.utils import load_data_to_predict, race_int, driver_numbers

MODEL_PATH='./model/extratrees_model.joblib'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

driver_numbers_reverse = {v: k for k, v in driver_numbers.items()}

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model

    except Exception as e:
        raise Exception("Model could not be loaded. {e}")

def main():
    
    parser = argparse.ArgumentParser(description="Predict F1 race results")
    parser.add_argument('--race', type=str, required=True, help='Race name (e.g., Monza)')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2023)')
    args = parser.parse_args()

    race = args.race
    year = args.year

    model = load_model()
    data,drivers = load_data_to_predict(race, year)

    if model and data is not None:
        predictions = model.predict(data)
        
        predicted_position = dict(zip(drivers,predictions))
        sorted_pred = dict(sorted(predicted_position.items(), key=lambda item: item[1]))
        print('='*10,f"Predictions for {race} 🏎️:",'='*10)
        for pos, (driver, pred) in enumerate(sorted_pred.items(), start=1):
            print(f"{pos}. {driver} (Predicted Position: {pred:.2f})")
    else:
        logger.error("Model or data is not available for predictions.")

if __name__ == "__main__":
    main()
