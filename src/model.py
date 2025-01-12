from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('model')

def create_pipeline():
    """Create model pipeline"""
    return Pipeline([
        ('regressor', XGBRegressor(
            random_state=Config.RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.1,
            n_jobs=-1
        ))
    ])

def train_model(pipeline, X_train, y_train, feature_names):
    """Train model with grid search CV"""
    try:
        logger.info("Starting model training with GridSearchCV...")
        
        # Update parameter grid for XGBoost
        param_grid = {
            'regressor__max_depth': [3, 4, 5, 6],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__n_estimators': [100, 200],
            'regressor__min_child_weight': [1, 3],
            'regressor__subsample': [0.8, 0.9],
            'regressor__colsample_bytree': [0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=Config.CV_FOLDS,
            n_jobs=-1,
            scoring='r2',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Get feature importance from the best model
        best_model = grid_search.best_estimator_
        xgb_model = best_model.named_steps['regressor']
        feature_importance = dict(zip(feature_names, xgb_model.feature_importances_))
        logger.info(f"Feature importance: {feature_importance}")
        
        # Save model
        import pickle
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        return best_model, feature_importance
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise