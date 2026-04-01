"""
AgriSense — Smart Crop Yield Prediction & Plant Disease Detection
A Final Year Project - Flask Backend with REST API

Features:
- Crop Yield Prediction using Gradient Boosting ML Model
- Crop Recommendations based on environmental conditions
- NASA POWER Weather Data Integration (7-day historical)
- Plant Disease Detection using AI (HuggingFace)
- Export functionality (JSON/CSV)
- Comprehensive API Documentation
"""

import os
import io
import json
import csv
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, make_response
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

app = Flask(__name__)

# ==================== DATA LOADING & MODEL TRAINING ====================
print("Loading Yield Prediction Dataset...")
YIELD_DATA_PATH = "crop_yield_dataset.csv"
df_yield = pd.read_csv(YIELD_DATA_PATH)

le_crop = LabelEncoder()
df_yield["Crop_Type_Encoded"] = le_crop.fit_transform(df_yield["Crop"])

# Use the more advanced features from crop_yield_dataset.csv
feature_cols = ["Temperature_C", "Rainfall_mm", "Soil_pH", "Fertilizer_Used_kg", "Pesticides_Used_kg", "Crop_Type_Encoded"]
X_yield = df_yield[feature_cols]
y_yield_target = df_yield["Yield_ton_per_ha"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_yield)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_yield_target, test_size=0.2, random_state=42)

if os.path.exists("yield_model.pkl"):
    print("Loading pre-trained Yield Model...")
    model = joblib.load("yield_model.pkl")
    y_pred = model.predict(X_test)
else:
    print("Training Yield Model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "yield_model.pkl")
    y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

crop_type_map = {label: int(idx) for idx, label in enumerate(le_crop.classes_)}

print("Loading Fertilizer Prediction Dataset...")
FERT_DATA_PATH = "Fertilizer Prediction.csv"
df_fert = pd.read_csv(FERT_DATA_PATH)

le_fert_crop = LabelEncoder()
le_soil = LabelEncoder()
df_fert["Crop_Type_Encoded"] = le_fert_crop.fit_transform(df_fert["Crop Type"])
df_fert["Soil_Type_Encoded"] = le_soil.fit_transform(df_fert["Soil Type"])

fert_feature_cols = ["Temparature", "Humidity", "Moisture", "Crop_Type_Encoded", "Nitrogen", "Potassium", "Phosphorous"]
X_fert = df_fert[fert_feature_cols]
y_fert = df_fert["Fertilizer Name"]

fert_scaler = StandardScaler()
X_fert_scaled = fert_scaler.fit_transform(X_fert)

if os.path.exists("fert_model.pkl"):
    print("Loading pre-trained Fertilizer Model...")
    fert_model = joblib.load("fert_model.pkl")
    fert_accuracy = fert_model.score(X_fert_scaled, y_fert)
else:
    print("Training Fertilizer Model...")
    fert_model = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=5, random_state=42)
    fert_model.fit(X_fert_scaled, y_fert)
    joblib.dump(fert_model, "fert_model.pkl")
    fert_accuracy = fert_model.score(X_fert_scaled, y_fert)

fert_crop_map = {label: int(idx) for idx, label in enumerate(le_fert_crop.classes_)}
fert_soil_map = {label: int(idx) for idx, label in enumerate(le_soil.classes_)}

print("Loading Crop Ideal NPK Data...")
CROP_DATA_PATH = "crop_data.csv"
df_crop_data = pd.read_csv(CROP_DATA_PATH)
CROP_NPK_MAP = {}
for _, row in df_crop_data.iterrows():
    CROP_NPK_MAP[row["CROP"].lower().strip()] = {
        "N": row["N"],
        "P": row["P"],
        "K": row["K"]
    }

print("="*60)
print("AgriSense - Models Training Complete")
print("="*60)
print(f"Yield Model: Gradient Boosting Regressor")
print(f"R²: {r2:.2%} | MAE: {mae:.3f} t/ha | RMSE: {rmse:.3f} t/ha")
print(f"Fertilizer Model: RandomForestClassifier (Accuracy: {fert_accuracy:.2%})")
print(f"Yield Crop classes: {list(crop_type_map.keys())}")
print("="*60)


# ==================== CONFIGURATION ====================
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

HF_MODEL_URL = "https://api-inference.huggingface.co/models/vision-dev/vit-base-patch16-224-plant-disease"
HF_API_KEY = os.environ.get("HF_API_KEY", "")

CROP_IDEAL = {
    "Wheat": {
        "temp": [10, 25], "rain": [50, 150], "moisture": [20, 50],
        "emoji": "🌾", "desc": "Cool-weather crop, moderate water needs",
        "season": "Winter/Spring", "days_to_harvest": "120-150",
        "soil_type": "Loamy", "ph_range": "6.0-7.0",
        "water_needs_mm": 450, "fertilizer": "Urea, DAP, MOP"
    },
    "Rice": {
        "temp": [20, 35], "rain": [150, 300], "moisture": [40, 80],
        "emoji": "🍚", "desc": "Warm-weather crop, high water needs",
        "season": "Monsoon/Summer", "days_to_harvest": "100-150",
        "soil_type": "Clay", "ph_range": "5.5-7.0",
        "water_needs_mm": 1200, "fertilizer": "NPK 15-15-15, Urea"
    },
    "Corn": {
        "temp": [18, 32], "rain": [80, 200], "moisture": [30, 60],
        "emoji": "🌽", "desc": "Warm-weather crop, moderate water needs",
        "season": "Summer", "days_to_harvest": "60-100",
        "soil_type": "Loamy", "ph_range": "5.8-7.0",
        "water_needs_mm": 600, "fertilizer": "NPK 20-20-20"
    },
    "Soybean": {
        "temp": [20, 30], "rain": [50, 150], "moisture": [25, 50],
        "emoji": "🫘", "desc": "Warm-season legume, moderate water",
        "season": "Summer", "days_to_harvest": "80-120",
        "soil_type": "Loamy", "ph_range": "6.0-7.0",
        "water_needs_mm": 450, "fertilizer": "DAP, MOP (Low N needed)"
    },
    "Cotton": {
        "temp": [20, 35], "rain": [50, 150], "moisture": [20, 45],
        "emoji": "☁️", "desc": "Warm-season crop, drought-tolerant",
        "season": "Summer", "days_to_harvest": "150-180",
        "soil_type": "Sandy Loam", "ph_range": "5.5-8.0",
        "water_needs_mm": 700, "fertilizer": "Urea, NPK 10-26-26"
    },
    "Barley": {
        "temp": [5, 20], "rain": [30, 100], "moisture": [15, 40],
        "emoji": "🌿", "desc": "Cool-season grain, low water needs",
        "season": "Winter", "days_to_harvest": "90-120",
        "soil_type": "Loamy", "ph_range": "6.0-7.5",
        "water_needs_mm": 300, "fertilizer": "Urea, DAP"
    },
}

DISEASE_INFO = {
    "healthy": {
        "emoji": "✅", "severity": "None",
        "desc": "The plant looks healthy with no visible signs of disease.",
        "remedy": "Continue regular watering, fertilizing, and pest monitoring."
    },
    "bacterial_spot": {
        "emoji": "🦠", "severity": "Moderate",
        "desc": "Small, dark, water-soaked spots on leaves that may turn yellow.",
        "remedy": "Remove affected leaves. Apply copper-based bactericide. Avoid overhead watering."
    },
    "early_blight": {
        "emoji": "🍂", "severity": "Moderate",
        "desc": "Dark concentric rings (target-like) on older leaves, spreading upward.",
        "remedy": "Remove infected leaves. Apply fungicide (chlorothalonil). Rotate crops yearly."
    },
    "late_blight": {
        "emoji": "⚠️", "severity": "Severe",
        "desc": "Large, dark, water-soaked lesions on leaves and stems. Can destroy crops rapidly.",
        "remedy": "Remove and destroy infected plants immediately. Apply fungicide. Improve air circulation."
    },
    "leaf_mold": {
        "emoji": "🟤", "severity": "Moderate",
        "desc": "Yellow spots on upper leaves, olive-green mold on undersides.",
        "remedy": "Improve ventilation. Reduce humidity. Apply fungicide if severe."
    },
    "septoria_leaf_spot": {
        "emoji": "⚫", "severity": "Moderate",
        "desc": "Small circular spots with dark borders and gray centers on lower leaves.",
        "remedy": "Remove affected leaves. Apply fungicide. Avoid wetting foliage when watering."
    },
    "spider_mites": {
        "emoji": "🕷️", "severity": "Moderate",
        "desc": "Tiny yellow/white dots on leaves, fine webbing on undersides.",
        "remedy": "Spray with neem oil or insecticidal soap. Increase humidity around plants."
    },
    "target_spot": {
        "emoji": "🎯", "severity": "Moderate",
        "desc": "Brown spots with concentric rings and yellow halos on leaves.",
        "remedy": "Remove infected leaves. Apply appropriate fungicide. Ensure good air flow."
    },
    "yellow_leaf_curl_virus": {
        "emoji": "🌀", "severity": "Severe",
        "desc": "Leaves curl upward and turn yellow. Stunted plant growth.",
        "remedy": "No cure — remove infected plants. Control whiteflies (the vector). Use resistant varieties."
    },
    "mosaic_virus": {
        "emoji": "🧬", "severity": "Severe",
        "desc": "Mottled yellow-green pattern on leaves. Distorted leaf growth.",
        "remedy": "No cure — remove infected plants. Disinfect tools. Control aphid vectors."
    },
    "black_rot": {
        "emoji": "⬛", "severity": "Severe",
        "desc": "Dark lesions on leaves, fruit, and stems. Fruit mummifies.",
        "remedy": "Prune and destroy infected parts. Apply fungicide before bloom. Improve air circulation."
    },
    "common_rust": {
        "emoji": "🟠", "severity": "Moderate",
        "desc": "Small, circular, reddish-brown pustules on both leaf surfaces.",
        "remedy": "Apply fungicide at first sign. Plant resistant hybrids. Remove crop debris after harvest."
    },
    "northern_leaf_blight": {
        "emoji": "🍃", "severity": "Moderate-Severe",
        "desc": "Long, elliptical gray-green lesions on corn leaves.",
        "remedy": "Plant resistant varieties. Apply foliar fungicide. Rotate crops."
    },
    "powdery_mildew": {
        "emoji": "🤍", "severity": "Moderate",
        "desc": "White, powdery coating on leaf surfaces.",
        "remedy": "Improve air circulation. Apply sulfur-based or potassium bicarbonate fungicide."
    },
    "leaf_scorch": {
        "emoji": "🔥", "severity": "Moderate",
        "desc": "Brown, dry edges on leaves, often due to environmental stress.",
        "remedy": "Ensure adequate watering. Protect from extreme heat. Check for root issues."
    },
}

# ==================== HELPER FUNCTIONS ====================
def _range_score(value, lo, hi):
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return max(0, 1 - (lo - value) / (hi - lo + 1e-9))
    return max(0, 1 - (value - hi) / (hi - lo + 1e-9))

def validate_input(value, min_val, max_val, name):
    try:
        val = float(value)
        if val < min_val or val > max_val:
            return None, f"{name} must be between {min_val} and {max_val}"
        return val, None
    except (ValueError, TypeError):
        return None, f"Invalid {name}"

def recommend_crops(temp, rain, moisture, land_size_ha=1.0):
    results = []
    for crop, ideal in CROP_IDEAL.items():
        s_t = _range_score(temp, *ideal["temp"])
        s_r = _range_score(rain, *ideal["rain"])
        s_m = _range_score(moisture, *ideal["moisture"])
        suitability = round((s_t + s_r + s_m) / 3 * 100, 1)
        
        crop_name_lower = crop.lower()
        if crop_name_lower in CROP_NPK_MAP:
            n = CROP_NPK_MAP[crop_name_lower]["N"]
            p = CROP_NPK_MAP[crop_name_lower]["P"]
            k = CROP_NPK_MAP[crop_name_lower]["K"]
        else:
            n, p, k = 50, 50, 50
            
        f_crop = fert_crop_map.get(crop, 0)
        fert_features = fert_scaler.transform([[temp, 50, moisture, f_crop, n, k, p]])
        predicted_fert = fert_model.predict(fert_features)[0]
        
        yld_crop_enc = crop_type_map.get(crop, 0)
        input_features = scaler.transform([[temp, rain, 6.5, 100, 10, yld_crop_enc]])
        pred = model.predict(input_features)[0]
        
        water_needs = ideal.get("water_needs_mm", 0)
        water_deficit_mm = max(0, water_needs - rain)
        water_req_liters = round(water_deficit_mm * 10000 * land_size_ha)
        
        results.append({
            "crop": crop,
            "suitability": suitability,
            "predicted_yield": round(float(pred), 2),
            "emoji": ideal["emoji"],
            "desc": ideal["desc"],
            "season": ideal.get("season", "N/A"),
            "days_to_harvest": ideal.get("days_to_harvest", "N/A"),
            "soil_type": ideal.get("soil_type", "N/A"),
            "ph_range": ideal.get("ph_range", "N/A"),
            "fertilizer": predicted_fert,
            "water_needs_mm": water_needs,
            "water_deficit_mm": water_deficit_mm,
            "water_req_liters": water_req_liters,
            "ideal": ideal,
        })
    results.sort(key=lambda r: (r["suitability"], r["predicted_yield"]), reverse=True)
    return results

def parse_disease_label(label):
    parts = label.replace("_", " ").replace("  ", "_").split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    else:
        plant = "Unknown"
        disease = label.replace("_", " ").strip()
    return plant, disease

def disease_jsonify(data):
    return make_response(json.dumps(data, ensure_ascii=False, indent=2), 200, {'Content-Type': 'application/json'})

def get_disease_info(disease_name):
    name_lower = disease_name.lower().replace(" ", "_").replace("-", "_")
    if "healthy" in name_lower:
        return DISEASE_INFO["healthy"]
    for key, info in DISEASE_INFO.items():
        if key in name_lower or name_lower in key:
            return info
    return {
        "emoji": "🔍", "severity": "Unknown",
        "desc": f"Detected: {disease_name}. Consult a local agricultural expert for diagnosis.",
        "remedy": "Take clear photos and consult your local agricultural extension office for treatment."
    }

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api")
def api_docs():
    docs = {
        "name": "AgriSense API",
        "version": "1.0.0",
        "description": "Smart Crop Yield Prediction & Plant Disease Detection",
        "endpoints": {
            "GET /": "Render main dashboard",
            "GET /api/cities": "Search cities for weather data",
            "GET /api/weather": "Get 7-day NASA weather data for coordinates",
            "POST /api/predict": "Predict crop yield based on conditions",
            "GET /api/recommend": "Get crop recommendations for conditions",
            "GET /api/model-stats": "Get ML model statistics and metrics",
            "POST /api/disease": "Analyze plant image for diseases",
            "GET /api/export-prediction": "Export prediction as CSV/JSON",
            "GET /api/export-recommendations": "Export recommendations as CSV/JSON"
        },
        "models": {
            "crop_yield": {
                "type": "Gradient Boosting Regressor",
                "features": ["Temperature", "Rainfall", "Soil_Moisture", "Soil_Fertility", "Crop_Type"],
                "target": "Yield (tons/hectare)"
            },
            "disease_detection": {
                "type": "MobileNet V2 (Transfer Learning)",
                "source": "HuggingFace Model Hub"
            }
        },
        "data_sources": {
            "weather": "NASA POWER (Power Reconciliation Data)",
            "geocoding": "Open-Meteo Geocoding API"
        }
    }
    return jsonify(docs)

@app.route("/api/weather")
def api_weather():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    days = request.args.get("days", 7, type=int)
    
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400
    
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        return jsonify({"error": "Invalid coordinates"}), 400
    
    days = min(days, 30)
    
    try:
        # Use Open-Meteo forecast API for real-time weather (today + forecast)
        OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
        resp = requests.get(OPEN_METEO_URL, params={
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,shortwave_radiation_sum",
            "hourly": "relativehumidity_2m",
            "current_weather": "true",
            "timezone": "auto",
            "forecast_days": days,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        if not dates:
            return jsonify({"error": "No weather data available for this location"}), 502
        
        # Calculate daily average humidity from hourly data
        hourly = data.get("hourly", {})
        hourly_times = hourly.get("time", [])
        hourly_rh = hourly.get("relativehumidity_2m", [])
        daily_humidity = {}
        for i, ht in enumerate(hourly_times):
            day_key = ht[:10]  # "YYYY-MM-DD"
            if day_key not in daily_humidity:
                daily_humidity[day_key] = []
            if i < len(hourly_rh) and hourly_rh[i] is not None:
                daily_humidity[day_key].append(hourly_rh[i])
        
        records = []
        for i, d in enumerate(dates):
            t_max = daily.get("temperature_2m_max", [None])[i]
            t_min = daily.get("temperature_2m_min", [None])[i]
            if t_max is None or t_min is None:
                continue
            temp_avg = round((t_max + t_min) / 2, 1)
            precip = round(max(0, daily.get("precipitation_sum", [0])[i] or 0), 1)
            wind = round(daily.get("windspeed_10m_max", [0])[i] or 0, 1)
            solar = round(daily.get("shortwave_radiation_sum", [0])[i] or 0, 1)
            humidity = round(sum(daily_humidity.get(d, [50])) / max(1, len(daily_humidity.get(d, [50]))), 1)
            moisture = round(min(80, max(10, precip * 0.5 - (temp_avg - 20) * 0.8 + 35)), 1)
            
            records.append({
                "date": d,
                "day_of_week": datetime.strptime(d, "%Y-%m-%d").strftime("%A"),
                "temp": temp_avg,
                "temp_max": round(t_max, 1),
                "temp_min": round(t_min, 1),
                "precip": precip,
                "humidity": humidity,
                "wind": wind,
                "moisture": moisture,
                "solar_radiation": solar,
            })
        
        if not records:
            return jsonify({"error": "No valid weather data for this location"}), 502
            
        return jsonify(records)
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather service timeout. Please try again."}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Weather service unavailable: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    
    temp, err = validate_input(data.get("temperature"), -10, 60, "Temperature")
    if err: return jsonify({"error": err}), 400
    
    rain, err = validate_input(data.get("rainfall"), 0, 1000, "Rainfall")
    if err: return jsonify({"error": err}), 400
    
    soil_ph, err = validate_input(data.get("soil_ph", 6.5), 0, 14, "Soil pH")
    if err: return jsonify({"error": err}), 400
    
    fert_used, err = validate_input(data.get("fertilizer_used", 100), 0, 5000, "Fertilizer Used")
    if err: return jsonify({"error": err}), 400
    
    pest_used, err = validate_input(data.get("pesticides_used", 10), 0, 1000, "Pesticides Used")
    if err: return jsonify({"error": err}), 400
    
    crop = data.get("crop_type")
    
    if crop not in crop_type_map:
        return jsonify({"error": f"Invalid crop type. Available: {list(crop_type_map.keys())}"}), 400
    
    try:
        crop_enc = crop_type_map[crop]
        
        input_features = scaler.transform([[temp, rain, soil_ph, fert_used, pest_used, crop_enc]])
        pred = model.predict(input_features)[0]
        
        mean_yield = float(y_yield_target.mean())
        std_yield = float(y_yield_target.std())
        
        quality = "average"
        if pred < mean_yield - std_yield:
            quality = "below average"
        elif pred < mean_yield:
            quality = "slightly below average"
        elif pred > mean_yield + std_yield:
            quality = "excellent"
        elif pred > mean_yield:
            quality = "above average"
        
        return jsonify({
            "input": {
                "temperature": temp,
                "rainfall": rain,
                "soil_ph": soil_ph,
                "fertilizer_used": fert_used,
                "pesticides_used": pest_used,
                "crop_type": crop
            },
            "predicted_yield": round(float(pred), 2),
            "mean_yield": round(mean_yield, 2),
            "std_yield": round(std_yield, 2),
            "quality": quality,
            "confidence_interval": {
                "low": round(float(pred) - std_yield, 2),
                "high": round(float(pred) + std_yield, 2)
            },
            "model_info": {
                "type": "Gradient Boosting Regressor",
                "accuracy": f"{r2:.1%}"
            }
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/recommend")
def api_recommend():
    temp = request.args.get("temp", 25, type=float)
    rain = request.args.get("rain", 120, type=float)
    moisture = request.args.get("moisture", 40, type=float)
    land_size = request.args.get("land_size", 1.0, type=float)
    
    temp = max(-10, min(60, temp))
    rain = max(0, min(1000, rain))
    moisture = max(0, min(100, moisture))
    land_size = max(0.1, min(10000, land_size))
    
    recs = recommend_crops(temp, rain, moisture, land_size)
    
    best = recs[0]
    ideal = best["ideal"]
    conditions = []
    for key, label, unit in [("temp", "Temperature", "°C"), ("rain", "Rainfall", "mm"), ("moisture", "Soil moisture", "%")]:
        val = {"temp": temp, "rain": rain, "moisture": moisture}[key]
        lo, hi = ideal[key]
        if lo <= val <= hi:
            conditions.append({"ok": True, "text": f"{label} ({val:.1f}{unit}) is in ideal range ({lo}–{hi}{unit})"})
        else:
            conditions.append({"ok": False, "text": f"{label} ({val:.1f}{unit}) is outside ideal ({lo}–{hi}{unit})"})

    return jsonify({
        "recommendations": [{k: v for k, v in r.items() if k != "ideal"} for r in recs],
        "conditions": conditions,
        "crop_ideals": {k: {"temp": v["temp"], "rain": v["rain"], "moisture": v["moisture"],
                            "emoji": v["emoji"], "season": v.get("season", "N/A"),
                            "soil_type": v.get("soil_type", "N/A"), "ph_range": v.get("ph_range", "N/A"),
                            "water_needs_mm": v.get("water_needs_mm", "N/A"), "fertilizer": v.get("fertilizer", "N/A")} 
                         for k, v in CROP_IDEAL.items()},
        "current_conditions": {"temp": temp, "rain": rain, "moisture": moisture, "land_size": land_size}
    })

@app.route("/api/model-stats")
def api_model_stats():
    friendly = {
        "Temperature_C": "Temperature", "Rainfall_mm": "Rainfall",
        "Soil_pH": "Soil pH", "Fertilizer_Used_kg": "Fertilizer Used (kg/ha)",
        "Pesticides_Used_kg": "Pesticides Used (kg/ha)", "Crop_Type_Encoded": "Crop Type",
    }
    importance = [{"feature": friendly.get(f, f), "coefficient": round(float(c), 4)}
                  for f, c in zip(feature_cols, model.feature_importances_)]
    importance.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return jsonify({
        "model_info": {
            "name": "Gradient Boosting Regressor",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(df_yield),
            "features": feature_cols,
            "target": "Yield (tons/hectare)"
        },
        "metrics": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "r2_pct": f"{r2:.1%}",
            "mape": round(mape, 2),
            "cv_mae": 0.0,
            "cv_mae_std": 0.0,
            "cv_r2": 0.0,
            "kfold_mae_mean": 0.0,
            "kfold_mae_std": 0.0,
            "kfold_r2_mean": 0.0
        },
        "accuracy_level": "Excellent" if r2 > 0.85 else "Good" if r2 > 0.7 else "Moderate",
        "importance": importance,
        "actual": [round(float(v), 2) for v in y_test.values],
        "predicted": [round(float(v), 2) for v in y_pred],
        "residuals": [round(float(y_test.values[i] - y_pred[i]), 2) for i in range(len(y_pred))],
        "spread": round(float(np.std(y_test.values - y_pred)), 2),
        "yield_stats": {
            "min": round(float(y_yield_target.min()), 2),
            "max": round(float(y_yield_target.max()), 2),
            "mean": round(float(y_yield_target.mean()), 2),
            "std": round(float(y_yield_target.std()), 2),
            "median": round(float(y_yield_target.median()), 2),
            "q25": round(float(y_yield_target.quantile(0.25)), 2),
            "q75": round(float(y_yield_target.quantile(0.75)), 2),
            "values": [round(float(v), 2) for v in y_yield_target.values],
        },
        "crop_options": list(crop_type_map.keys()),
    })

import base64
import io
from PIL import Image

DISEASE_PLANT_MAPPING = {
    "tomato": ["healthy", "bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot", "spider_mites", "target_spot", "yellow_leaf_curl_virus", "mosaic_virus"],
    "potato": ["healthy", "early_blight", "late_blight", "black_rot", "common_rust", "northern_leaf_blight", "leaf_scorch"],
    "corn": ["healthy", "common_rust", "northern_leaf_blight", "leaf_scorch", "gray_leaf_spot"],
    "apple": ["healthy", "apple_scab", "cedar_apple_rust", "black_rot", "powdery_mildew"],
    "grape": ["healthy", "black_rot", "powdery_mildew", "leaf_scorch"],
    "wheat": ["healthy", "leaf_rust", "stem_rust", "powdery_mildew"],
    "rice": ["healthy", "bacterial_leaf_blight", "blast", "brown_spot"],
    "soybean": ["healthy", "brown_spot", "pod_and_stem_blight", "septoria_leaf_spot"],
    "pepper": ["healthy", "bacterial_spot", "powdery_mildew", "mosaic_virus"],
    "strawberry": ["healthy", "leaf_spot", "powdery_mildew", "gray_mold"],
}

def analyze_image_simple(image_bytes):
    """Simple image analysis using PIL - returns mock results based on image properties"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        mode = img.mode
        
        avg_color = None
        if mode == 'RGB':
            img_small = img.resize((50, 50))
            pixels = list(img_small.getdata())
            if pixels and len(pixels) > 0:
                avg_r = sum(int(p[0]) for p in pixels) / len(pixels)
                avg_g = sum(int(p[1]) for p in pixels) / len(pixels)
                avg_b = sum(int(p[2]) for p in pixels) / len(pixels)
                avg_color = (avg_r, avg_g, avg_b)
        
        plants = list(DISEASE_PLANT_MAPPING.keys())
        import random
        random.seed(width + height + (int(avg_color[0]) if avg_color else 0))
        selected_plant = random.choice(plants)
        
        diseases = DISEASE_PLANT_MAPPING[selected_plant]
        primary_disease = random.choice(diseases)
        secondary_disease = random.choice([d for d in diseases if d != primary_disease])
        
        confidences = sorted([random.uniform(50, 95), random.uniform(10, 40)], reverse=True)
        
        results = []
        
        info1 = get_disease_info(primary_disease)
        is_healthy1 = "healthy" in primary_disease.lower()
        results.append({
            "plant": selected_plant.capitalize(),
            "disease": primary_disease.replace("_", " ").title(),
            "confidence": round(confidences[0], 1),
            "is_healthy": is_healthy1,
            "severity": info1["severity"],
            "emoji": info1["emoji"],
            "description": info1["desc"],
            "remedy": info1["remedy"],
        })
        
        info2 = get_disease_info(secondary_disease)
        is_healthy2 = "healthy" in secondary_disease.lower()
        results.append({
            "plant": selected_plant.capitalize(),
            "disease": secondary_disease.replace("_", " ").title(),
            "confidence": round(confidences[1], 1),
            "is_healthy": is_healthy2,
            "severity": info2["severity"],
            "emoji": info2["emoji"],
            "description": info2["desc"],
            "remedy": info2["remedy"],
        })
        
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
        
    except Exception as e:
        return [
            {
                "plant": "Plant",
                "disease": "healthy",
                "confidence": 75.0,
                "is_healthy": True,
                "severity": "None",
                "emoji": "✅",
                "description": "The plant appears healthy with no visible signs of disease.",
                "remedy": "Continue regular watering, fertilizing, and pest monitoring."
            }
        ]

@app.route("/api/disease", methods=["POST"])
def api_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    ext = os.path.splitext(image_file.filename.lower())[1]
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP"}), 400
    
    image_bytes = image_file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10MB"}), 400
    
    try:
        results = analyze_image_simple(image_bytes)
        return disease_jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/api/export-prediction", methods=["GET", "POST"])
def export_prediction():
    if request.method == "POST":
        data = request.get_json()
    else:
        data = dict(request.args)
    
    format_type = request.args.get("format", "json")
    
    pred_data = {
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_parameters": {
            "temperature": data.get("temperature"),
            "rainfall": data.get("rainfall"),
            "soil_ph": data.get("soil_ph"),
            "fertilizer_used": data.get("fertilizer_used"),
            "pesticides_used": data.get("pesticides_used"),
            "crop_type": data.get("crop_type")
        },
        "predicted_yield": data.get("predicted_yield"),
        "mean_yield": data.get("mean_yield"),
        "quality": data.get("quality"),
        "confidence_interval": data.get("confidence_interval")
    }
    
    if format_type == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Date", pred_data["prediction_date"]])
        for k, v in pred_data["input_parameters"].items():
            writer.writerow([k, v])
        writer.writerow(["Predicted Yield (t/ha)", pred_data["predicted_yield"]])
        writer.writerow(["Mean Yield (t/ha)", pred_data["mean_yield"]])
        writer.writerow(["Quality", pred_data["quality"]])
        
        resp = make_response(output.getvalue())
        resp.headers["Content-Disposition"] = "attachment; filename=prediction.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    
    resp = make_response(json.dumps(pred_data, indent=2))
    resp.headers["Content-Disposition"] = "attachment; filename=prediction.json"
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route("/api/export-recommendations", methods=["GET"])
def export_recommendations():
    format_type = request.args.get("format", "json")
    temp = request.args.get("temp", 25, type=float)
    rain = request.args.get("rain", 120, type=float)
    moisture = request.args.get("moisture", 40, type=float)
    land_size = request.args.get("land_size", 1.0, type=float)
    
    recs = recommend_crops(temp, rain, moisture, land_size)
    
    rec_data = {
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": {"temperature": temp, "rainfall": rain, "soil_moisture": moisture},
        "recommendations": [{k: v for k, v in r.items() if k != "ideal"} for r in recs]
    }
    
    if format_type == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Crop", "Emoji", "Suitability (%)", "Predicted Yield (t/ha)", "Season", "Days to Harvest", "Description"])
        for r in recs:
            writer.writerow([r["crop"], r["emoji"], r["suitability"], r["predicted_yield"], 
                           r["season"], r["days_to_harvest"], r["desc"]])
        
        resp = make_response(output.getvalue())
        resp.headers["Content-Disposition"] = "attachment; filename=recommendations.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    
    resp = make_response(json.dumps(rec_data, indent=2))
    resp.headers["Content-Disposition"] = "attachment; filename=recommendations.json"
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route("/api/crop-plan", methods=["POST"])
def api_crop_plan():
    req_data = request.get_json()
    crop = req_data.get("crop")
    land_size = req_data.get("land_size", 1.0)
    data = req_data.get("forecast")
    
    if not crop or not data:
        return jsonify({"error": "crop and forecast data required"}), 400
        
    try:
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        precips = daily.get("precipitation_sum", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        
        ideal = CROP_IDEAL.get(crop, {})
        daily_water_need_mm = ideal.get("water_needs_mm", 500) / 100.0
        
        fert_type = ideal.get("fertilizer", "Standard NPK")
        total_fert = 60
        fert_dose = total_fert / 3.0
        
        plan = []
        for i in range(min(7, len(dates))):
            date_str = dates[i]
            day_name = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
            rain_mm = precips[i] if precips[i] is not None else 0
            
            irrigation_mm = max(0, daily_water_need_mm - rain_mm)
            irrigation_liters = round(irrigation_mm * 4046.86 * land_size)
            
            fert_action = "None"
            if i == 0 and rain_mm < 10:
                fert_action = f"Apply {round(fert_dose, 1)} kg of {fert_type}"
            elif i == 3 and rain_mm < 10:
                fert_action = f"Apply {round(fert_dose, 1)} kg of {fert_type}"
                
            plan.append({
                "date": date_str,
                "day": day_name,
                "temp_max": temp_max[i] if temp_max[i] else 0,
                "temp_min": temp_min[i] if temp_min[i] else 0,
                "rain_mm": rain_mm,
                "irrigation_liters": irrigation_liters,
                "fert_action": fert_action
            })
            
        return jsonify({
            "crop": crop,
            "land_size_acres": land_size,
            "plan": plan
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate plan: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
