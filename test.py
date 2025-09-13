from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
import os
import uvicorn
from dateutil.relativedelta import relativedelta

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
    remaining_days = (warranty_end_date - today).days
    remaining_months = max(0, remaining_days / 30.44)

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

@app.get("/")
def home():
    return {"message": "Railway Lifecycle Monitoring ML API is running"}

def get_failure_cause(row, feature_cols, model):
    causes = []
    SAFE_VOLTAGE = 230
    SAFE_VIBRATION = 5
    MAX_PRODUCT_AGE = 365 * 10
    WARRANTY_SOON = 30
    INSPECTION_OVERDUE = 180

    voltage = row.get('voltage', None)
    if voltage is not None and voltage != "unknown":
        try:
            if float(voltage) > SAFE_VOLTAGE:
                causes.append(f"Voltage too high ({voltage})")
            elif float(voltage) < 0.8 * SAFE_VOLTAGE:
                causes.append(f"Voltage too low ({voltage})")
        except Exception:
            pass

    vibration = row.get('vibration', None)
    if vibration is not None and vibration != "unknown":
        try:
            if float(vibration) > SAFE_VIBRATION:
                causes.append(f"Vibration above safe limit ({vibration})")
        except Exception:
            pass

    product_age = row.get('productAge', None)
    if product_age is not None and product_age != "unknown":
        try:
            if float(product_age) > MAX_PRODUCT_AGE:
                causes.append(f"Product age very high ({product_age} days)")
        except Exception:
            pass

    warranty_months = row.get('warrantyMonths', None)
    install_date = row.get('installDate_parsed', None)
    manufacture_date = row.get('manufactureDate_parsed', None)
    warranty_info = calculate_warranty_details(warranty_months, install_date, manufacture_date)
    if warranty_info.get("status") in ["critical", "upcoming"]:
        causes.append(f"Warranty expiring soon ({warranty_info.get('remaining_days', '?')} days left)")
    elif warranty_info.get("status") == "expired":
        causes.append("Warranty expired")

    last_insp_col = None
    for c in row.index if hasattr(row, "index") else []:
        if "inspection" in c.lower() and "date" in c.lower():
            last_insp_col = c
            break
    if last_insp_col:
        try:
            last_insp = pd.to_datetime(row[last_insp_col])
            days_since = (datetime.now(timezone.utc) - last_insp).days
            if days_since > INSPECTION_OVERDUE:
                causes.append(f"Inspection overdue ({days_since} days since last inspection)")
        except Exception:
            pass

    if not causes:
        feature_importances = model.feature_importances_
        feature_vals = []
        for feat in feature_cols:
            try:
                val = row.get(feat, None)
                feature_vals.append(val)
            except Exception:
                feature_vals.append(None)
        cause_scores = []
        for i, feat in enumerate(feature_cols):
            try:
                col_vals = model.X_train_[:, i] if hasattr(model, "X_train_") else None
                median = np.median(col_vals) if col_vals is not None else None
                val = feature_vals[i]
                if median is not None and val is not None and not pd.isna(val):
                    score = abs(val - median) * feature_importances[i]
                else:
                    score = 0
                cause_scores.append(score)
            except Exception:
                cause_scores.append(0)
        top_indices = np.argsort(cause_scores)[::-1][:2]
        for idx in top_indices:
            feat = feature_cols[idx]
            val = feature_vals[idx]
            causes.append(f"{feat}={val}")
    return ", ".join(causes) if causes else "N/A"

@app.post("/predict")
def predict(request: PredictRequest):
    df = fetch_and_prepare_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="No data fetched from DATA_URL or data empty")

    target_col = 'currentStatus'
    feature_cols = get_feature_columns(df)

    if target_col not in df.columns or df[target_col].nunique() < 2:
        return {
            "message": "Not enough variation in 'currentStatus' for ML training. Check data.",
            "unique_statuses": df[target_col].unique().tolist() if target_col in df.columns else []
        }

    X = df[feature_cols]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    model.X_train_ = X.values

    df_pred = pd.DataFrame()
    if request.productId:
        pid = str(request.productId).strip().lower()
        if 'productId_norm' in df.columns:
            df_pred = df[df['productId_norm'] == pid]
        if df_pred.empty:
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
            "detail": "No data found for given filter(s). Possible causes: mismatch in id string, different field name, nested list storage.",
            "requested_productId": request.productId,
            "requested_lotId": request.lotId,
            "sample_productIds_first_20": sample_product_ids,
            "sample_lotIds_first_20": sample_lot_ids,
            "available_columns": df.columns.tolist()
        }

    X_pred = df_pred[feature_cols]
    X_pred = X_pred.fillna(0)

    alerts = {
        "failure_risk": [],
        "inspection": [],
        "depot": [],
        "trend": [],
        "vendor": [],
        "location": [],
        "recommendation": [],
        "warranty": []
    }

    warranty_details = []

    # --- Inspection Alerts for individual products ---
    for _, row in df_pred.iterrows():
        product_id = row.get("productId", "unknown")
        lot_id = row.get("lotId", "unknown")
        # Look for inspection/recommendation columns
        for c in row.index:
            if "inspection" in c.lower() and "recommendation" in c.lower() and row[c] not in ["unknown", None, ""]:
                alerts["inspection"].append({
                    "level": "Recommendation",
                    "message": f"🔎 Inspection Alert → Product {product_id} in Lot {lot_id}: Recommendation - {row[c]}"
                })
            if "inspection" in c.lower() and "result" in c.lower() and row[c] not in ["unknown", None, ""]:
                alerts["inspection"].append({
                    "level": "Result",
                    "message": f"🔎 Inspection Alert → Product {product_id} in Lot {lot_id}: Result - {row[c]}"
                })

    # --- Depot Alerts ---
    if 'depotId' in df.columns and not df_pred.empty:
        depot_id = df_pred.iloc[0].get('depotId', None)
        if depot_id and depot_id != "unknown":
            depot_products = df[df['depotId'] == depot_id]
            if len(depot_products) > 0:
                fail_rate = depot_products['currentStatus'].mean() if 'currentStatus' in depot_products.columns else None
                if fail_rate is not None and fail_rate > 0.5:
                    alerts["depot"].append({
                        "level": "High",
                        "message": f"🏭 Depot Alert → Depot {depot_id} has high failure rate ({fail_rate:.2f})"
                    })

    # --- Trend Alerts (simple: if product failure rate is increasing over time) ---
    if 'createdAt' in df.columns and 'currentStatus' in df.columns:
        df_sorted = df.sort_values('createdAt')
        window = min(10, len(df_sorted))
        if window >= 5:
            recent = df_sorted.tail(window)
            earlier = df_sorted.head(window)
            recent_fail = recent['currentStatus'].mean()
            earlier_fail = earlier['currentStatus'].mean()
            if recent_fail > earlier_fail + 0.1:
                alerts["trend"].append({
                    "level": "Rising",
                    "message": (
                        f"📈 Trend Alert → Failure rate is rising: {earlier_fail:.2f} → {recent_fail:.2f}. "
                        "This means that more products are failing in recent records compared to earlier ones. "
                        "Consider investigating possible causes or scheduling preventive maintenance."
                    )
                })

    try:
        y_proba = model.predict_proba(X_pred)
    except Exception:
        y_proba = np.zeros((len(X_pred), 2))

    for i, prob in enumerate(y_proba):
        risk = prob[1] if prob.shape[0] == 2 or prob.shape[0] > 1 and len(prob) > 1 else prob.max()
        product_id = df_pred.iloc[i].get('productId', f'row_{i}')
        lot_id = df_pred.iloc[i].get('lotId', 'unknown')
        try:
            risk_val = float(risk)
        except Exception:
            risk_val = 0.0

        cause = get_failure_cause(df_pred.iloc[i], feature_cols, model)
        risk_percent = int(round(risk_val * 100))

        if risk_val > 0.8:
            alerts["failure_risk"].append({
                "level": "High",
                "message": f"🔴 Failure Risk Alert → Product {product_id} in {lot_id} likely to fail (risk {risk_percent}%). Cause: {cause}"
            })
        elif risk_val > 0.5:
            alerts["failure_risk"].append({
                "level": "Medium",
                "message": f"🟠 Medium Risk → Product {product_id} in {lot_id} needs inspection (risk {risk_percent}%). Cause: {cause}"
            })
        else:
            alerts["failure_risk"].append({
                "level": "Low",
                "message": f"🟢 Low Risk → Product {product_id} in {lot_id} is operating normally (risk {risk_percent}%). Cause: {cause}"
            })

    for _, row in df_pred.iterrows():
        product_id = row.get("productId", "unknown")
        lot_id = row.get("lotId", "unknown")
        warranty_months = row.get("warrantyMonths", None)
        install_date = row.get('installDate_parsed', None)
        manufacture_date = row.get('manufactureDate_parsed', None)
        warranty_info = calculate_warranty_details(warranty_months, install_date, manufacture_date)

        warranty_details.append({
            "productId": product_id,
            "lotId": lot_id,
            "warrantyMonths": warranty_months,
            "installDate": str(install_date) if install_date is not None else "unknown",
            "manufactureDate": str(manufacture_date) if manufacture_date is not None else "unknown",
            **warranty_info
        })

        if warranty_info["status"] == "expired":
            alerts["warranty"].append({
                "level": "Expired",
                "message": f"⚠️ Warranty Expired → Product {product_id} in Lot {lot_id} expired on {warranty_info['end_date']}"
            })
        elif warranty_info["status"] == "critical":
            alerts["warranty"].append({
                "level": "Critical",
                "message": f"🔴 Warranty Alert → Product {product_id} in Lot {lot_id} warranty expiring in {warranty_info['remaining_days']} days (ends {warranty_info['end_date']})"
            })
        elif warranty_info["status"] == "upcoming":
            alerts["warranty"].append({
                "level": "Upcoming",
                "message": f"🟠 Warranty Alert → Product {product_id} in Lot {lot_id} warranty expiring in {warranty_info['remaining_days']} days (ends {warranty_info['end_date']})"
            })
        elif warranty_info["status"] == "valid":
            alerts["warranty"].append({
                "level": "Safe",
                "message": f"🟢 Warranty Valid → Product {product_id} in Lot {lot_id} has {warranty_info['remaining_days']} days remaining (ends {warranty_info['end_date']})"
            })
        else:
            alerts["warranty"].append({
                "level": "Unknown",
                "message": f"⚠️ {warranty_info['message']} for {product_id} in {lot_id}"
            })

    if 'vendorId' in df.columns and 'currentStatus' in df.columns:
        vendor_failures = df.groupby('vendorId')['currentStatus'].mean()
        threshold = vendor_failures.mean() + vendor_failures.std()
        high_risk_vendors = vendor_failures[vendor_failures > threshold]
        for vendor, risk in high_risk_vendors.items():
            alerts["vendor"].append({
                "level": "High",
                "message": f"🟠 Vendor Alert → Vendor {vendor} has high failure rate ({risk:.2f})"
            })

    if 'latitude' in df.columns and 'longitude' in df.columns and 'currentStatus' in df.columns:
        df['latlon'] = df['latitude'].astype(str) + "," + df['longitude'].astype(str)
        location_failures = df.groupby('latlon')['currentStatus'].mean()
        if not location_failures.empty:
            threshold = location_failures.mean() + location_failures.std()
            high_risk_locations = location_failures[location_failures > threshold]
            for loc, risk in high_risk_locations.items():
                alerts["location"].append({
                    "level": "High",
                    "message": f"🟠 Location Alert → Location {loc} has high failure rate ({risk:.2f})"
                })
        else:
            high_risk_locations = pd.Series(dtype=float)

    if 'latlon' in df.columns and 'vibration' in df.columns and 'voltage' in df.columns and not high_risk_locations.empty:
        for loc in high_risk_locations.index:
            loc_df = df[df['latlon'] == loc]
            safe_vibration = loc_df[loc_df['currentStatus'] == 0]['vibration'].median() if not loc_df[loc_df['currentStatus'] == 0].empty else None
            safe_voltage = loc_df[loc_df['currentStatus'] == 0]['voltage'].median() if not loc_df[loc_df['currentStatus'] == 0].empty else None
            if safe_vibration is not None and safe_voltage is not None:
                alerts["recommendation"].append({
                    "location": loc,
                    "message": f"🟢 To reduce failures at {loc}, keep vibration below {safe_vibration:.2f} and voltage near {safe_voltage:.2f}"
                })

    return {
        "feature_importances": {feat: float(importance) for feat, importance in zip(feature_cols, model.feature_importances_)},
        "alerts": alerts,
        "warranty_details": warranty_details,
        "used_features": feature_cols,
        "matched_rows_count": len(df_pred)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("test:app", host="0.0.0.0", port=port, reload=True)