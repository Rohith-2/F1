import joblib
import logging
import argparse
import warnings

from src.utils import load_data_to_predict, driver_numbers, MODEL_PATH, convert

warnings.filterwarnings("ignore")

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
    parser.add_argument('--race', type=int, required=True, help='Race no. (e.g., 2,3)')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2023)')
    parser.add_argument('--verbose',type=bool, default=False, help='Enable verbose logging')
    args = parser.parse_args()

    race = args.race
    year = args.year

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger('fastf1').setLevel(logging.ERROR)

    model = load_model()
    logger.info("Loading Data for Predicting...")
    data,drivers = load_data_to_predict(race, year)
    logger.info("Data Loaded. Making Predictions...")

    if model and data is not None:
        predictions = model.predict(data)
        
        predicted_position = dict(zip(drivers,predictions))
        sorted_pred = dict(sorted(predicted_position.items(), key=lambda item: item[1]))
        sorted_pred = {k: convert(v) for k, v in sorted_pred.items()}
        print('\n')
        print('='*10,f"Predictions for Race No. {race} 🏎️:",'='*10)
        for pos, (driver, pred) in enumerate(sorted_pred.items(), start=1):
            print(f"{pos}. {driver} (Predicted Laptime: {pred})")
    else:
        logger.error("Model or data is not available for predictions.")

if __name__ == "__main__":
    main()
