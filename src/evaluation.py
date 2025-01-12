from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import json
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('evaluation')

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate model performance"""
    try:
        # Make predictions
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': float(r2_score(y_train, pred_train)),
            'test_r2': float(r2_score(y_test, pred_test)),
            'train_rmse': float(np.sqrt(mean_squared_error(np.exp(y_train), np.exp(pred_train)))),
            'test_rmse': float(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred_test)))),
            'train_mae': float(mean_absolute_error(np.exp(y_train), np.exp(pred_train))),
            'test_mae': float(mean_absolute_error(np.exp(y_test), np.exp(pred_test)))
        }
        
        # Get feature importance from XGBoost model
        xgb_model = model.named_steps['regressor']
        # Convert numpy float32 to Python float
        importance_values = [float(x) for x in xgb_model.feature_importances_]
        feature_importance = dict(zip(feature_names, importance_values))
        
        # Save metrics
        with open(Config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save feature importance
        with open(Config.FEATURE_IMPORTANCE_PATH, 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        logger.info("Model evaluation completed and saved")
        return metrics, feature_importance
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise