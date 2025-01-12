from src.data_preparation import load_and_prepare_data
from src.model import create_pipeline, train_model
from src.evaluation import evaluate_model
from utils.logger import setup_logger

logger = setup_logger('train')

def main():
    try:
        # Load dan prepare data
        logger.info("Loading and preparing data...")
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

        # Create dan train model
        logger.info("Creating and training model...")
        pipeline = create_pipeline()
        model, feature_importance = train_model(pipeline, X_train, y_train, feature_names)

        # Evaluasi model
        logger.info("Evaluating model...")
        metrics, _ = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        
        logger.info("Training completed successfully")
        logger.info(f"Test R2 Score: {metrics['test_r2']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()