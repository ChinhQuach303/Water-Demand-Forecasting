import pandas as pd
import numpy as np
import os
import mlflow
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from config import setup_logging, DATA_DIR, set_seed

logger = setup_logging(__name__)

def detailed_evaluation(y_true, y_pred, name):
    y_true_arr = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()
    diff = y_true_arr - y_pred_arr
    total_samples = len(y_true_arr)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    shortage_mask = diff > 0
    shortage_count = np.sum(shortage_mask)
    shortage_pct = (shortage_count / total_samples) * 100
    avg_shortage_vol = np.sum(diff[shortage_mask]) / total_samples
    
    surplus_mask = diff < 0
    avg_surplus_vol = np.sum(np.abs(diff[surplus_mask])) / total_samples
    
    metrics = {
        f'{name}_MAE': mae,
        f'{name}_R2': r2,
        f'{name}_Shortage_Pct': shortage_pct,
        f'{name}_Avg_Shortage': avg_shortage_vol,
        f'{name}_Avg_Surplus': avg_surplus_vol
    }
    
    logger.info(f"📊 RESULTS FOR: {name}")
    logger.info(f"📈 MAE              : {mae:,.2f}")
    logger.info(f"📈 R2 Score         : {r2:.4f}")
    logger.info(f"🔴 Shortage %       : {shortage_pct:.2f}%")
    
    return metrics

def main():
    set_seed()
    logger.info("Starting Evaluation Step...")
    
    # Connect to MLflow (assuming local run or connect to server)
    # In production, we'd fetch the latest model from Registry or use run ID passed as arg.
    # For this script, we'll try to find the latest run or just load the local artifacts for simplicity of the flow.
    # To correspond with the Plan, let's load from the local artifact paths created by train.py until we have a full server.
    
    try:
        model = joblib.load("model_hybrid.pkl")
        safety_layer = joblib.load("safety_layer.pkl")
        features = joblib.load("features_list.pkl")
    except FileNotFoundError:
        logger.error("❌ Model artifacts not found. Run train.py first.")
        return

    test_df = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
    X_test, y_test = test_df[features], test_df['Demand']
    
    # Predict
    raw_pred = model.predict(X_test)
    final_pred = safety_layer.predict(raw_pred, test_df)
    
    # Evaluate
    metrics = detailed_evaluation(y_test, final_pred, "TEST")
    
    # Log to MLflow - ideally we should resume the run or create a new evaluation run linked to it.
    # For now, we just print/log locally.
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()
