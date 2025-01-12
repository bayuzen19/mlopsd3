import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('data_preparation')

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    try:
        # Load data
        logger.info("Loading data from CSV...")
        df = pd.read_csv(Config.DATA_PATH)
        
        # Split features and target
        X = df[Config.FEATURE_COLUMNS]
        y = np.log(df[Config.TARGET_COLUMN])
        
        # Save feature names
        feature_names = X.columns.tolist()
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info("Data preparation completed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise