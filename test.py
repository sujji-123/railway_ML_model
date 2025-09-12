from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import uvicorn

app = FastAPI(title="Railway Lifecycle Monitoring ML API")

DATA_URL = "https://railway-yrlm.onrender.com/api/products"

class PredictRequest(BaseModel):
    productId: Optional[str] = None
    lotId: Optional[str] = None

def fetch_and_prepare_data() -> pd.DataFrame:
    response = requests.get(DATA_URL)
    data = response.json()
    products = data.get("products", [])
    all_records = []
    for record in products:
        flat = pd.json_normalize(record, sep='_')
        for col in flat.columns:
            flat[col] = flat[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
        all_records.append(flat)
    df = pd.concat(all_records, ignore_index=True)
    df.fillna("unknown", inplace=True)

    # Convert categorical columns to numeric codes (except IDs, dates, vendorId)
    for col in df.columns:
        if df[col].dtype == object and col not in ['manufactureDate', 'installDate', 'productId', 'lotId', 'vendorId']:
            df[col] = df[col].astype('category').cat.codes

    # Normalize numeric columns (except IDs and target)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['productId', 'lotId', 'currentStatus']:
            max_val = df[col].max()
            min_val = df[col].min()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    # Feature Engineering
    today = pd.to_datetime(datetime.now()).tz_localize(None)
    if 'manufactureDate' in df.columns:
        df['manufactureDate'] = pd.to_datetime(df['manufactureDate'], errors='coerce').dt.tz_localize(None)
        df['productAge'] = (today - df['manufactureDate']).dt.days
        df['productAge'] = df['productAge'].fillna(-1)
    if 'installDate' in df.columns:
        df['installDate'] = pd.to_datetime(df['installDate'], errors='coerce').dt.tz_localize(None)
        df['installDuration'] = (today - df['installDate']).dt.days
        df['installDuration'] = df['installDuration'].fillna(-1)
    return df

def get_feature_columns(df: pd.DataFrame):
    # Add all relevant features for ML
    possible_features = [
        'productType', 'warrantyMonths', 'productAge', 'installDuration',
        'voltage', 'vibration', 'temperature', 'latitude', 'longitude', 'vendorId'
    ]
    # Only use features that exist and are numeric/categorical
    feature_cols = [col for col in possible_features if col in df.columns]
    # Add any other numeric columns not excluded
    exclude = {'productId', 'lotId', 'currentStatus'}
    feature_cols += [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude and col not in feature_cols]
    return list(set(feature_cols))

@app.get("/")
def home():
    return {"message": "Railway Lifecycle Monitoring ML API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    df = fetch_and_prepare_data()
    target_col = 'currentStatus'
    feature_cols = get_feature_columns(df)

    if target_col not in df.columns or df[target_col].nunique() < 2:
        return {
            "message": "Not enough variation in 'currentStatus' for ML training. Check data.",
            "unique_statuses": df[target_col].unique().tolist() if target_col in df.columns else []
        }

    # Train on ALL data
    X = df[feature_cols]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Filter for prediction/alerts only
    if request.productId:
        df_pred = df[df['productId'] == request.productId]
    elif request.lotId:
        df_pred = df[df['lotId'] == request.lotId]
    else:
        df_pred = df

    if df_pred.empty:
        raise HTTPException(status_code=404, detail="No data found for given filter(s)")

    X_pred = df_pred[feature_cols]
    y_proba = model.predict_proba(X_pred)

    # AI Alerts
    alerts = {
        "failure_risk": [],
        "inspection": [],
        "depot": [],
        "trend": [],
        "vendor": [],
        "location": [],
        "recommendation": []
    }

    # Failure Risk Alerts (per product)
    for i, prob in enumerate(y_proba):
        risk = prob[1] if len(prob) > 1 else prob[0]
        product_id = df_pred.iloc[i]['productId']
        lot_id = df_pred.iloc[i]['lotId']
        if risk > 0.8:
            alerts["failure_risk"].append({
                "level": "High",
                "message": f"🔴 Failure Risk Alert → Product {product_id} in {lot_id} likely to fail (risk {risk:.2f})"
            })
        elif risk > 0.5:
            alerts["failure_risk"].append({
                "level": "Medium",
                "message": f"🟠 Medium Risk → Product {product_id} in {lot_id} needs inspection (risk {risk:.2f})"
            })
        else:
            alerts["failure_risk"].append({
                "level": "Low",
                "message": f"🟢 Low Risk → Product {product_id} in {lot_id} is operating normally (risk {risk:.2f})"
            })

    # Vendor Risk Alerts
    if 'vendorId' in df.columns and 'currentStatus' in df.columns:
        vendor_failures = df.groupby('vendorId')['currentStatus'].mean()
        threshold = vendor_failures.mean() + vendor_failures.std()
        high_risk_vendors = vendor_failures[vendor_failures > threshold]
        for vendor, risk in high_risk_vendors.items():
            alerts["vendor"].append({
                "level": "High",
                "message": f"🟠 Vendor Alert → Vendor {vendor} has high failure rate ({risk:.2f})"
            })

    # Location Risk Alerts (by latitude/longitude cluster or unique location)
    if 'latitude' in df.columns and 'longitude' in df.columns and 'currentStatus' in df.columns:
        # Example: group by rounded lat/lon for "location"
        df['latlon'] = df['latitude'].astype(str) + "," + df['longitude'].astype(str)
        location_failures = df.groupby('latlon')['currentStatus'].mean()
        threshold = location_failures.mean() + location_failures.std()
        high_risk_locations = location_failures[location_failures > threshold]
        for loc, risk in high_risk_locations.items():
            alerts["location"].append({
                "level": "High",
                "message": f"🟠 Location Alert → Location {loc} has high failure rate ({risk:.2f})"
            })

    # Parameter Recommendations (for high-risk locations)
    if 'latlon' in df.columns and 'vibration' in df.columns and 'voltage' in df.columns:
        for loc in high_risk_locations.index:
            loc_df = df[df['latlon'] == loc]
            # Suggest median vibration/voltage where failures are low
            safe_vibration = loc_df[loc_df['currentStatus'] == 0]['vibration'].median() if not loc_df[loc_df['currentStatus'] == 0].empty else None
            safe_voltage = loc_df[loc_df['currentStatus'] == 0]['voltage'].median() if not loc_df[loc_df['currentStatus'] == 0].empty else None
            if safe_vibration is not None and safe_voltage is not None:
                alerts["recommendation"].append({
                    "location": loc,
                    "message": f"🟢 To reduce failures at {loc}, keep vibration below {safe_vibration:.2f} and voltage near {safe_voltage:.2f}"
                })

    # other alerts (inspection, depot, trend) can be added similarly based on business logic

    return {
        "feature_importances": {feat: float(importance) for feat, importance in zip(feature_cols, model.feature_importances_)},
        "alerts": alerts,
        "used_features": feature_cols
    }


# 👇 Added section to handle local + Render startup automatically
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT, fallback to 8000 locally
    uvicorn.run("test:app", host="127.0.0.1", port=port, reload=True)
