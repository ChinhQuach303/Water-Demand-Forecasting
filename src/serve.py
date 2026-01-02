import pandas as pd
import numpy as np
import joblib
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import make_asgi_app, Counter, Histogram
from config import setup_logging

logger = setup_logging("Serving")

app = FastAPI(title="Water Demand Forecasting API")

# Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("request_count", "Total prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of predictions")

# Load Models
try:
    # In a real Docker env, these would be mounted or fetched from MLflow
    model = joblib.load("model_hybrid.pkl")
    safety_layer = joblib.load("safety_layer.pkl")
    features = joblib.load("features_list.pkl")
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    model = None
    safety_layer = None

class InputData(BaseModel):
    # Minimal schema based on features
    # 'PWSID_enc', 'Month', 'Year', 'Is_Summer_Peak', 'lag_1', 'lag_12', 'diff_12', 'CDD'
    # In a real scenario, we might accept raw inputs and do the feature engineering here.
    # For this simplified E2E, we accept the precomputed features.
    PWSID_enc: int
    Month: int
    Year: int
    Is_Summer_Peak: int
    lag_1: float
    lag_12: float
    diff_12: float
    CDD: float
    # Context data for safety_layer
    # It needs 'lag_1', 'lag_12', 'Month', 'PWSID_enc' which are above.
    
    # We might need to construct a DataFrame for the model
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: List[InputData]):
    REQUEST_COUNT.inc()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([d.dict() for d in data])
        
        # Ensure column order
        X = df[features]
        
        # Raw Predict
        with PREDICTION_LATENCY.time():
            raw_pred = model.predict(X)
            # Safety Layer
            # Safety layer needs the dataframe context
            final_pred = safety_layer.predict(raw_pred, df)
            
        return {"predictions": final_pred.tolist()}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
