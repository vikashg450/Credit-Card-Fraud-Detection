import os
import pickle
import json
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="SENTINEL AI Fraud Detection Engine")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "templates")

# Ensure static directories exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files (plots, css, js)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables for models and scaler
lr_model = None
xgb_model = None
scaler = None
test_data = None
feature_columns = None
metrics_data = None

# Correlation signs of variables with Class (based on EDA correlation analysis)
# Positive means higher value drives towards fraud, negative means lower value drives towards fraud
CORR_SIGNS = {
    "Time": -1, "Amount": 1,
    "V1": -1, "V2": 1, "V3": -1, "V4": 1, "V5": -1, "V6": -1, "V7": -1, "V8": 1, "V9": -1, "V10": -1,
    "V11": 1, "V12": -1, "V13": -1, "V14": -1, "V15": -1, "V16": -1, "V17": -1, "V18": -1, "V19": 1, "V20": 1,
    "V21": 1, "V22": 1, "V23": -1, "V24": -1, "V25": 1, "V26": 1, "V27": 1, "V28": 1
}

@app.on_event("startup")
def startup_event():
    global lr_model, xgb_model, scaler, test_data, feature_columns, metrics_data
    
    print("Initializing Sentinel API...")
    
    # 1. Load scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    else:
        print(f"Warning: Scaler not found at {scaler_path}")
        
    # 2. Load models
    lr_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    if os.path.exists(lr_path):
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
        print("Logistic Regression model loaded.")
    else:
        print(f"Warning: LR model not found at {lr_path}")
        
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        print("XGBoost model loaded.")
    else:
        print(f"Warning: XGBoost model not found at {xgb_path}")
        
    # 3. Load test split data for random sampling
    split_data_path = os.path.join(MODELS_DIR, "split_data.npz")
    if os.path.exists(split_data_path):
        test_data = np.load(split_data_path, allow_pickle=True)
        feature_columns = list(test_data["columns"])
        print("Test split data loaded.")
    else:
        print(f"Warning: Test split data not found at {split_data_path}")
        
    # 4. Load saved metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        print("Metrics file loaded.")
    else:
        print(f"Warning: Metrics file not found at {metrics_path}")

# Request / Response Schemas
class FeatureInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

class PredictionRequest(BaseModel):
    model_name: str  # "Logistic Regression" or "XGBoost"
    features: FeatureInput

class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability: float
    explanation: dict

# Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/api/metrics")
def get_metrics():
    if metrics_data:
        return metrics_data
    raise HTTPException(status_code=404, detail="Metrics not found")

@app.get("/api/random-sample")
def get_random_sample(class_type: int = 0):
    if test_data is None:
        raise HTTPException(status_code=500, detail="Test dataset not loaded")
        
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    
    # Filter indices matching the requested class
    matching_indices = np.where(y_test == class_type)[0]
    
    if len(matching_indices) == 0:
        raise HTTPException(status_code=404, detail=f"No transactions found for Class {class_type}")
        
    random_idx = random.choice(matching_indices)
    scaled_row = X_test[random_idx]
    
    # Reconstruct dictionary
    row_dict = {feature_columns[i]: float(scaled_row[i]) for i in range(len(feature_columns))}
    
    # Unscale Time and Amount for intuitive user input
    if scaler is not None:
        # Time index is 0, Amount index is 29 (last column of X)
        time_scaled = row_dict["Time"]
        amount_scaled = row_dict["Amount"]
        
        # Reshape for inverse transform
        raw_vals = scaler.inverse_transform([[time_scaled, amount_scaled]])[0]
        row_dict["Time"] = float(raw_vals[0])
        row_dict["Amount"] = float(raw_vals[1])
        
    return row_dict

@app.post("/api/predict", response_model=PredictionResponse)
def predict_transaction(payload: PredictionRequest):
    global lr_model, xgb_model, scaler
    
    model_name = payload.model_name
    features = payload.features.dict()
    
    # Choose Model
    if model_name == "XGBoost" and xgb_model is not None:
        model = xgb_model
    elif model_name == "Logistic Regression" and lr_model is not None:
        model = lr_model
    else:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded or unavailable")
        
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler is not initialized")
        
    # Scale Time and Amount
    time_raw = features["Time"]
    amount_raw = features["Amount"]
    
    scaled_vals = scaler.transform([[time_raw, amount_raw]])[0]
    
    # Construct scaled feature vector in correct order
    scaled_features = {}
    for col in feature_columns:
        if col == "Time":
            scaled_features[col] = scaled_vals[0]
        elif col == "Amount":
            scaled_features[col] = scaled_vals[1]
        else:
            scaled_features[col] = features[col]
            
    # Convert to numpy array of shape (1, 30)
    input_vector = np.array([[scaled_features[col] for col in feature_columns]])
    
    # Predict
    prob = float(model.predict_proba(input_vector)[0, 1])
    pred = int(model.predict(input_vector)[0])
    
    # Explain prediction locally
    # We will use Logistic Regression coefficients as our base local explanation metric,
    # as they represent clear directional gradients.
    # contribution = scaled_value * coefficient
    explanation = {}
    
    if lr_model is not None:
        coefs = lr_model.coef_[0]
        for idx, col in enumerate(feature_columns):
            val = scaled_features[col]
            coef = coefs[idx]
            # Contribution represents how much this feature drives the log-odds towards fraud
            contrib = val * coef
            explanation[col] = float(contrib)
    else:
        # Fallback explanation using correlation sign and feature value
        for col in feature_columns:
            val = scaled_features[col]
            sign = CORR_SIGNS.get(col, 1)
            # Roughly estimate driving factor
            contrib = val * sign
            explanation[col] = float(contrib)
            
    return PredictionResponse(
        model_name=model_name,
        prediction=pred,
        probability=prob,
        explanation=explanation
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI development server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
