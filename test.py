from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel, validator
  
from typing import Optional, List, Dict, Any
import requests  
import pandas as pd  
import numpy as np  
from datetime import datetime, timezone  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import StandardScaler
import os  
import uvicorn  
from dateutil.relativedelta import relativedelta  
import json
from collections import defaultdict
from enum import Enum
import math
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone
import pandas as pd

app = FastAPI(title="Railway Lifecycle Monitoring ML API")  

DATA_URL = "https://railway-yrlm.onrender.com/api/products"

class Condition(str, Enum):
    GOOD = "Good"
    WORN = "Worn"
    DAMAGED = "Damaged"

# class VibrationLevel(str, Enum):
#     LOW = "Low"
#     HIGH = "High"
#     CRITICAL = "Critical"

class InspectionRecommendationRequest(BaseModel):
    condition: Condition
    voltage_reading: float
    vibration_level: float   # now numeric instead of Enum
    past_inspection_data: Optional[List[Dict[str, Any]]] = None

    # --- Auto-capitalize condition only (vibration is numeric now) ---
    @validator("condition", pre=True, always=True)
    def normalize_condition(cls, v):
        if isinstance(v, str):
            return v.capitalize()
        return v

class PredictRequest(BaseModel):  
    productId: Optional[str] = None  
    lotId: Optional[str] = None

class ProductDetailsRequest(BaseModel):
    productId: str

# Global variables for model and scaler
model = None
scaler = StandardScaler()
feature_columns = []

def fetch_and_prepare_data() -> pd.DataFrame:
    # Your existing implementation
    response = requests.get(DATA_URL)  
    data = response.json()  
    products = data.get("products", [])  
    all_records = []  
    for record in products:  
        flat = pd.json_normalize(record, sep='_')  
        tms = record.get("tms", {})  
        if tms and "installedDate" in tms:  
            flat["tms_installedDate"] = tms["installedDate"]  
        all_records.append(flat)  
    if not all_records:  
        return pd.DataFrame()  
    df = pd.concat(all_records, ignore_index=True)  
    df.fillna("unknown", inplace=True)  

    if 'manufactureDate' in df.columns:  
        df['manufactureDate_parsed'] = pd.to_datetime(df['manufactureDate'], errors='coerce').dt.tz_localize(None)  
    else:  
        df['manufactureDate_parsed'] = pd.NaT  

    install_cols = []  
    if 'tms_installedDate' in df.columns:  
        install_cols.append('tms_installedDate')  
    if 'installDate' in df.columns:  
        install_cols.append('installDate')  
    if 'tmsRecordId_installedDate' in df.columns:  
        install_cols.append('tmsRecordId_installedDate')  
    if install_cols:  
        chosen = install_cols[0]  
        df['installDate_parsed'] = pd.to_datetime(df[chosen], errors='coerce').dt.tz_localize(None)  
    else:  
        df['installDate_parsed'] = pd.NaT  

    if 'productId' in df.columns:  
        df['productId_str'] = df['productId'].astype(str)  
        df['productId_norm'] = df['productId_str'].str.strip().str.lower()  
    if 'lotId' in df.columns:  
        df['lotId_str'] = df['lotId'].astype(str)  
        df['lotId_norm'] = df['lotId_str'].str.strip().str.lower()  

    # Exclude inspection result/recommendation columns from categorical encoding  
    inspection_exclude = []  
    for col in df.columns:  
        col_lower = col.lower()  
        if ("inspection" in col_lower or "insp" in col_lower) and ("result" in col_lower or "recommendation" in col_lower):  
            inspection_exclude.append(col)  

    for col in df.columns:  
        if (  
            df[col].dtype == object and  
            col not in [  
                'manufactureDate', 'installDate', 'tms_installedDate',  
                'productId', 'lotId', 'vendorId',  
                'productId_str', 'productId_norm', 'lotId_str', 'lotId_norm',  
                'manufactureDate_parsed', 'installDate_parsed'  
            ] + inspection_exclude  
        ):  
            try:  
                df[col] = df[col].astype('category').cat.codes  
            except Exception:  
                pass  

    for col in df.select_dtypes(include=[np.number]).columns:  
        if col not in ['productId', 'lotId', 'currentStatus', 'warrantyMonths']:  
            max_val = df[col].max()  
            min_val = df[col].min()  
            if max_val != min_val:  
                df[col] = (df[col] - min_val) / (max_val - min_val)  

    today = pd.to_datetime(datetime.now(timezone.utc)).tz_localize(None)  
    df['productAge'] = (today - df['manufactureDate_parsed']).dt.days  
    df['productAge'] = df['productAge'].fillna(-1)  
    df['installDuration'] = (today - df['installDate_parsed']).dt.days  
    df['installDuration'] = df['installDuration'].fillna(-1)  

    return df

def get_feature_columns(df: pd.DataFrame):  
    possible_features = [  
        'productType', 'warrantyMonths', 'productAge', 'installDuration',  
        'voltage', 'vibration', 'temperature', 'latitude', 'longitude', 'vendorId'  
    ]  
    feature_cols = [col for col in possible_features if col in df.columns]  
    exclude = {'productId', 'lotId', 'currentStatus'}  
    feature_cols += [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude and col not in feature_cols]  
    return list(set(feature_cols))

def calculate_warranty_details(warranty_months, install_date, manufacture_date):
    """
    Calculates the warranty status, remaining days, and end date for a product.
    This corrected version provides an accurate calculation for remaining days.
    """
    today = datetime.now(timezone.utc)

    if warranty_months in (None, "unknown", "") or pd.isna(warranty_months):
        return {
            "status": "unknown",
            "remaining_days": None,
            "remaining_months": None,
            "end_date": None,
            "message": "Warranty information not available"
        }

    try:
        warranty_months_int = int(round(float(warranty_months)))
    except (ValueError, TypeError):
        return {
            "status": "unknown",
            "remaining_days": None,
            "remaining_months": None,
            "end_date": None,
            "message": "Invalid warranty months value"
        }

    start_date = None
    if pd.notnull(install_date) and str(install_date) != "unknown":
        dt = pd.to_datetime(install_date)
        start_date = dt if dt.tzinfo else dt.tz_localize(timezone.utc)
    elif pd.notnull(manufacture_date) and str(manufacture_date) != "unknown":
        dt = pd.to_datetime(manufacture_date)
        start_date = dt if dt.tzinfo else dt.tz_localize(timezone.utc)

    if start_date is None:
        return {
            "status": "unknown",
            "remaining_days": None,
            "remaining_months": None,
            "end_date": None,
            "message": "No install or manufacture date available"
        }

    warranty_end_date = start_date + relativedelta(months=warranty_months_int)
    
    # --- FIX STARTS HERE ---
    # Calculate the precise difference in seconds and convert to days
    time_difference = warranty_end_date - today
    # Use math.ceil to round up, so any remaining time counts as a full day.
    # max(0, ...) ensures we don't show negative days for expired warranties.
    remaining_days = math.ceil(max(0, time_difference.total_seconds() / (24 * 60 * 60)))
    # --- FIX ENDS HERE ---
    
    remaining_months = max(0, remaining_days / 30.44) # Using average days in a month

    if remaining_days <= 0:
        return {
            "status": "expired",
            "remaining_days": 0,
            "remaining_months": 0,
            "end_date": warranty_end_date.strftime("%Y-%m-%d"),
            "message": f"Warranty expired on {warranty_end_date.strftime('%Y-%m-%d')}"
        }
    elif remaining_days <= 30:
        return {
            "status": "critical",
            "remaining_days": remaining_days,
            "remaining_months": round(remaining_months, 1),
            "end_date": warranty_end_date.strftime("%Y-%m-%d"),
            "message": f"Warranty expiring in {remaining_days} days ({round(remaining_months, 1)} months)"
        }
    elif remaining_days <= 90:
        return {
            "status": "upcoming",
            "remaining_days": remaining_days,
            "remaining_months": round(remaining_months, 1),
            "end_date": warranty_end_date.strftime("%Y-%m-%d"),
            "message": f"Warranty expiring in {remaining_days} days ({round(remaining_months, 1)} months)"
        }
    else:
        return {
            "status": "valid",
            "remaining_days": remaining_days,
            "remaining_months": round(remaining_months, 1),
            "end_date": warranty_end_date.strftime("%Y-%m-%d"),
            "message": f"Warranty valid for {remaining_days} days ({round(remaining_months, 1)} months)"
        }

def analyze_inspection_data(row) -> Dict[str, Any]:
    """Analyze inspection data and generate insights"""
    inspection_results = []
    replace_count = 0
    repair_count = 0
    ok_count = 0
    
    # Look for inspection-related columns
    for col_name, value in row.items():
        if "inspection" in col_name.lower() and "recommendation" in col_name.lower():
            if value and str(value).lower() == "replace":
                replace_count += 1
                inspection_results.append({"type": "Replace", "source": col_name})
            elif value and str(value).lower() == "repair":
                repair_count += 1
                inspection_results.append({"type": "Repair", "source": col_name})
            elif value and str(value).lower() in ["ok", "good", "satisfactory"]:
                ok_count += 1
                inspection_results.append({"type": "OK", "source": col_name})
        
        # Also check for inspection results
        if "inspection" in col_name.lower() and "result" in col_name.lower():
            if value and "fail" in str(value).lower():
                inspection_results.append({"type": "Failure", "source": col_name})
    
    # Generate summary message
    summary_parts = []
    if ok_count > 0:
        summary_parts.append(f"{ok_count} OK")
    if repair_count > 0:
        summary_parts.append(f"{repair_count} Repair")
    if replace_count > 0:
        summary_parts.append(f"{replace_count} Replace")
    
    summary = "Mixed inspection results" if len(summary_parts) > 1 else "Inspection results"
    summary += f": {', '.join(summary_parts)}" if summary_parts else ": keep inspecting frequently for early detection"
    
    # Risk assessment based on inspection results
    risk_level = "Low"
    if replace_count >= 2:
        risk_level = "High"
        summary += ". Multiple 'Replace' recommendations indicate high failure probability."
    elif replace_count == 1 or repair_count >= 2:
        risk_level = "Medium"
        summary += ". Needs attention in next maintenance cycle."
    
    return {
        "summary": summary,
        "risk_level": risk_level,
        "details": inspection_results,
        "counts": {"ok": ok_count, "repair": repair_count, "replace": replace_count}
    }

def generate_ai_insights(product_data, risk_score, warranty_info, inspection_analysis):
    """Generate AI-powered insights based on all available data"""
    insights = []
    recommendations = []
    
    # Warranty insights
    if warranty_info["status"] == "valid":
        insights.append(f"Warranty valid with {warranty_info['remaining_days']} days remaining")
        
        # If warranty is valid but inspection shows issues
        if inspection_analysis["risk_level"] in ["Medium", "High"]:
            insights.append(f"Early risk detected: {inspection_analysis['counts']['replace']} 'Replace' recommendations despite valid warranty")
    elif warranty_info["status"] in ["critical", "upcoming"]:
        insights.append(f"Warranty expiring soon: {warranty_info['remaining_days']} days remaining")
    elif warranty_info["status"] == "expired":
        insights.append("Warranty has expired")
    
    # Risk score insights
    if risk_score > 80:
        insights.append(f"High failure risk score: {risk_score}%")
        recommendations.append("Consider immediate replacement")
    elif risk_score > 60:
        insights.append(f"Medium failure risk score: {risk_score}%")
        recommendations.append("Schedule inspection within 15 days")
    else:
        insights.append(f"Low failure risk score: {risk_score}%")
    
    # Inspection insights
    insights.append(inspection_analysis["summary"])
    
    # Environmental factors
    if 'vibration' in product_data and product_data['vibration'] not in [None, "unknown"]:
        try:
            vibration = float(product_data['vibration'])
            if vibration > 5.0:
                insights.append(f"High vibration detected: {vibration}")
                recommendations.append("Check mounting and alignment")
        except (ValueError, TypeError):
            pass
    
    if 'voltage' in product_data and product_data['voltage'] not in [None, "unknown"]:
        try:
            voltage = float(product_data['voltage'])
            if voltage > 250 or voltage < 200:
                insights.append(f"Voltage out of optimal range: {voltage}V")
                recommendations.append("Check power supply stability")
        except (ValueError, TypeError):
            pass
    
    # Generate predictive replacement date
    if warranty_info["end_date"] and inspection_analysis["risk_level"] == "High":
        try:
            end_date = datetime.strptime(warranty_info["end_date"], "%Y-%m-%d")
            # If high risk, recommend replacement 1 month before warranty expires
            recommended_date = end_date - relativedelta(months=1)
            insights.append(f"Recommended replacement by: {recommended_date.strftime('%Y-%m-%d')}")
        except (ValueError, TypeError):
            pass
    
    # If no specific recommendations, add general ones
    if not recommendations:
        if risk_score > 50:
            recommendations.append("Increase inspection frequency to every 15 days")
        else:
            recommendations.append("Continue with regular maintenance schedule")
    
    return {
        "insights": insights,
        "recommendations": recommendations,
        "risk_score": risk_score,
        "warranty_status": warranty_info["status"],
        "inspection_summary": inspection_analysis["summary"]
    }

def train_model():
    """Train the ML model with the latest data"""
    global model, feature_columns, scaler
    
    df = fetch_and_prepare_data()
    if df.empty:
        return False
    
    target_col = 'currentStatus'
    feature_columns = get_feature_columns(df)
    
    if target_col not in df.columns or df[target_col].nunique() < 2:
        return False
    
    X = df[feature_columns]
    y = df[target_col]
    
    # Scale the features
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return True
def classify_voltage(voltage: float) -> str:
    if voltage < 200 or voltage > 250:
        return "Replace"
    elif 200 <= voltage < 210 or 240 < voltage <= 250:
        return "Repair"
    else:
        return "OK"

def classify_vibration(vibration: float) -> str:
    if vibration > 7.0:
        return "Replace"
    elif 5.0 <= vibration <= 7.0:
        return "Repair"
    else:
        return "OK"


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the application starts"""
    train_model()

@app.get("/")
def home():
    return {"message": "Railway Lifecycle Monitoring ML API is running"}

@app.post("/inspection-recommendation")
def inspection_recommendation(request: InspectionRecommendationRequest):
    """
    Generate inspection recommendation based on equipment condition, voltage reading, 
    vibration level, and past inspection data.
    """
    # Check for expired equipment based on past inspection data
    if request.past_inspection_data:
        for inspection in request.past_inspection_data:
            if inspection.get("recommendation") == "Expired" or inspection.get("condition") == "Expired":
                return {
                    "recommendation": "Expired",
                    "message": "Product expired, needs immediate replacement"
                }
    
    # --- Updated: Use numeric voltage & vibration classification ---
    voltage_status = classify_voltage(request.voltage_reading)
    vibration_status = classify_vibration(request.vibration_level)

    # If either requires Replace
    if "Replace" in [voltage_status, vibration_status]:
        return {"recommendation": "Replace", "message": "High risk detected due to unsafe voltage or vibration"}

    # If either requires Repair
    if "Repair" in [voltage_status, vibration_status]:
        return {"recommendation": "Repair", "message": "Moderate issue detected, schedule repair"}

    # Check for damaged condition
    if request.condition == Condition.DAMAGED:
        return {
            "recommendation": "Replace",
            "message": "Replace within next 30 days due to damaged condition"
        }
    
    # Check for worn condition
    if request.condition == Condition.WORN:
        return {
            "recommendation": "Repair",
            "message": "Equipment needs repair due to worn condition"
        }
    
    # If all checks pass, equipment is OK
    return {
        "recommendation": "OK",
        "message": "Equipment is in good condition with normal readings"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    global model, feature_columns
    
    if model is None:
        if not train_model():
            raise HTTPException(status_code=500, detail="Failed to train model. Check data availability.")
    
    df = fetch_and_prepare_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="No data fetched from DATA_URL or data empty")
    
    df_pred = pd.DataFrame()
    if request.productId:
        pid = str(request.productId).strip().lower()
        if 'productId_norm' in df.columns:
            df_pred = df[df['productId_norm'] == pid]
        if df_pred.empty:
            # Try to find the product using other methods (your existing code)
            candidate_cols = [c for c in df.columns if 'products' in c.lower() or 'udm' in c.lower() or 'productid' in c.lower()]
            for c in candidate_cols:
                try:
                    mask = df[c].astype(str).str.lower().str.contains(pid, na=False)
                    if mask.any():
                        df_pred = df[mask]
                        break
                except Exception:
                    continue
        if df_pred.empty:
            string_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith('category')]
            for c in string_cols:
                try:
                    mask = df[c].astype(str).str.lower().str.contains(pid, na=False)
                    if mask.any():
                        df_pred = df[mask]
                        break
                except Exception:
                    continue

    elif request.lotId:
        lid = str(request.lotId).strip().lower()
        if 'lotId_norm' in df.columns:
            df_pred = df[df['lotId_norm'] == lid]
        if df_pred.empty:
            string_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith('category')]
            for c in string_cols:
                try:
                    mask = df[c].astype(str).str.lower().str.contains(lid, na=False)
                    if mask.any():
                        df_pred = df[mask]
                        break
                except Exception:
                    continue
    else:
        df_pred = df

    if df_pred.empty:
        sample_product_ids = df['productId'].astype(str).head(20).tolist() if 'productId' in df.columns else []
        sample_lot_ids = df['lotId'].astype(str).head(20).tolist() if 'lotId' in df.columns else []
        return {
            "detail": "No data found for given filter(s).",
            "requested_productId": request.productId,
            "requested_lotId": request.lotId,
            "sample_productIds_first_20": sample_product_ids,
            "sample_lotIds_first_20": sample_lot_ids
        }

    X_pred = df_pred[feature_columns]
    X_pred = X_pred.fillna(0)
    X_pred_scaled = scaler.transform(X_pred)

    try:
        y_proba = model.predict_proba(X_pred_scaled)
    except Exception:
        y_proba = np.zeros((len(X_pred), 2))

    results = []
    for i, (_, row) in enumerate(df_pred.iterrows()):
        risk_score = int(round(y_proba[i][1] * 100)) if y_proba.shape[1] > 1 else 0
        
        # Get warranty information
        warranty_months = row.get("warrantyMonths", None)
        install_date = row.get('installDate_parsed', None)
        manufacture_date = row.get('manufactureDate_parsed', None)
        warranty_info = calculate_warranty_details(warranty_months, install_date, manufacture_date)
        
        # Analyze inspection data
        inspection_analysis = analyze_inspection_data(row)
        
        # Generate AI insights
        ai_insights = generate_ai_insights(row, risk_score, warranty_info, inspection_analysis)
        
        results.append({
            "productId": row.get("productId", "unknown"),
            "lotId": row.get("lotId", "unknown"),
            "risk_score": risk_score,
            "warranty_info": warranty_info,
            "inspection_analysis": inspection_analysis,
            "ai_insights": ai_insights
        })

    return {"predictions": results}

@app.post("/product-details")
def get_product_details(request: ProductDetailsRequest):
    """Get detailed information for a specific product"""
    global model, feature_columns
    
    if model is None:
        if not train_model():
            raise HTTPException(status_code=500, detail="Failed to train model. Check data availability.")
    
    df = fetch_and_prepare_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")
    
    # Find the specific product
    pid = str(request.productId).strip().lower()
    product_data = None
    
    if 'productId_norm' in df.columns:
        product_match = df[df['productId_norm'] == pid]
        if not product_match.empty:
            product_data = product_match.iloc[0].to_dict()
    
    if product_data is None:
        # Try other methods to find the product (similar to predict endpoint)
        candidate_cols = [c for c in df.columns if 'product' in c.lower() or 'udm' in c.lower()]
        for c in candidate_cols:
            try:
                mask = df[c].astype(str).str.lower().str.contains(pid, na=False)
                if mask.any():
                    product_data = df[mask].iloc[0].to_dict()
                    break
            except Exception:
                continue
    
    if product_data is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Prepare features for prediction
    feature_dict = {col: product_data.get(col, 0) for col in feature_columns}
    X_pred = pd.DataFrame([feature_dict])[feature_columns]
    X_pred = X_pred.fillna(0)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Get prediction
    try:
        y_proba = model.predict_proba(X_pred_scaled)
        risk_score = int(round(y_proba[0][1] * 100)) if y_proba.shape[1] > 1 else 0
    except Exception:
        risk_score = 0
    
    # Get warranty information
    warranty_months = product_data.get("warrantyMonths", None)
    install_date = product_data.get('installDate_parsed', None)
    manufacture_date = product_data.get('manufactureDate_parsed', None)
    warranty_info = calculate_warranty_details(warranty_months, install_date, manufacture_date)
    
    # Analyze inspection data
    inspection_analysis = analyze_inspection_data(product_data)
    
    # Generate AI insights
    ai_insights = generate_ai_insights(product_data, risk_score, warranty_info, inspection_analysis)
    
    # Prepare response
    response = {
        "productId": product_data.get("productId", "unknown"),
        "lotId": product_data.get("lotId", "unknown"),
        "productType": product_data.get("productType", "unknown"),
        "manufactureDate": product_data.get("manufactureDate", "unknown"),
        "currentStatus": product_data.get("currentStatus", "unknown"),
        "risk_score": risk_score,
        "warranty_info": warranty_info,
        "inspection_analysis": inspection_analysis,
        "ai_insights": ai_insights
    }
    
    # Add location data if available
    if 'latitude' in product_data and 'longitude' in product_data:
        response["location"] = {
            "latitude": product_data.get("latitude"),
            "longitude": product_data.get("longitude")
        }
    
    return response

@app.post("/retrain-model")
def retrain_model():
    """Force retraining of the ML model"""
    success = train_model()
    return {"success": success, "message": "Model retrained" if success else "Failed to retrain model"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
