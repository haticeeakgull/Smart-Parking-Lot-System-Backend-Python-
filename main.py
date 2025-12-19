import os
import requests
import pandas as pd
import numpy as np
import joblib
import firebase_admin
import openmeteo_requests
import requests_cache
from datetime import datetime
from typing import AsyncGenerator, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from retry_requests import retry
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 1. .env dosyasından gizli anahtarları yükle
load_dotenv()

# --- Global Değişkenler ---
model = None
scaler = None
FIREBASE_APP = None
DB = None
PARK_MAP = {}

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "firebase_admin_key.json")

HOLIDAYS = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25',
]

# --- Yardımcı Fonksiyon: Doluluk Tahmini ---
async def calculate_occupancy_ratio(park_id: str, target_time: datetime):
    park_info = PARK_MAP.get(park_id)
    if not park_info:
        return None

    dt = target_time
    hour = dt.hour
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    is_holiday = 1 if dt.strftime("%Y/%m/%d") in HOLIDAYS else 0

    weather = get_weather_forecast(
        dt,
        lat=park_info.get("latitude", 38.7223),
        lon=park_info.get("longitude", -9.1393),
    )

    df_input = pd.DataFrame([{
        "hour": hour, "dayofweek": dayofweek, "is_weekend": is_weekend,
        "is_holiday": is_holiday, "park_id_encoded": park_info["encoded_id"],
        "max_capacity": park_info["capacity"], "temperature": weather["temperature"],
        "precipitation": weather["precipitation"], "wind_speed": weather["wind_speed"],
        "pressure": weather["pressure"],
    }])

    scale_cols = ["hour", "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])
    
    FEATURES = ["hour", "dayofweek", "is_weekend", "is_holiday", "park_id_encoded", 
                "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]

    ratio = float(model.predict(df_input[FEATURES])[0])
    return max(0.0, min(1.0, ratio))

# --- Hava Durumu ---
def get_weather_forecast(target_time: datetime, lat: float, lon: float):
    cache = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "surface_pressure"],
        "forecast_days": 3,
    }

    try:
        response = client.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        df = pd.DataFrame({
            "date": pd.to_datetime(hourly.Time(), unit="s", utc=True),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "pressure": hourly.Variables(3).ValuesAsNumpy(),
        })
        target_time_utc = pd.to_datetime(target_time).tz_localize("UTC") if target_time.tzinfo is None else target_time
        row = df.iloc[(df["date"] - target_time_utc).abs().argsort()[:1]]
        return {
            "temperature": float(row["temperature"].values[0]),
            "precipitation": float(row["precipitation"].values[0]),
            "wind_speed": float(row["wind_speed"].values[0]),
            "pressure": float(row["pressure"].values[0]),
        }
    except:
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 10.0, "pressure": 1013.0}

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model, scaler, FIREBASE_APP, DB, PARK_MAP
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        FIREBASE_APP = firebase_admin.initialize_app(cred)
        DB = firestore.client()
        print("✅ Firebase bağlantısı kuruldu")

        docs = DB.collection("otoparklar").stream()
        for doc in docs:
            data = doc.to_dict()
            park_id = data.get("park_id")
            if park_id:
                PARK_MAP[park_id] = {
                    "encoded_id": data.get("park_id_encoded"),
                    "capacity": data.get("max_capacity"),
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                }
        print(f"✅ {len(PARK_MAP)} otopark yüklendi")

        model = joblib.load("retrained_occupancy_model.joblib")
        scaler = joblib.load("retrained_standard_scaler.joblib")
        print("✅ Model ve Scaler yüklendi")
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
    yield
    print("🛑 Uygulama kapanıyor")

app = FastAPI(title="Akıllı Otopark Öneri API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Request Modelleri ---
class PredictionRequest(BaseModel):
    park_id: str
    prediction_time: datetime

class RecommendationRequest(BaseModel):
    target_lat: float
    target_lon: float
    arrival_time: datetime

# --- Endpoint 1: Tekil Tahmin ---
@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):
    ratio = await calculate_occupancy_ratio(request.park_id, request.prediction_time)
    if ratio is None: raise HTTPException(status_code=404, detail="Otopark bulunamadı")
    
    park_info = PARK_MAP[request.park_id]
    cars = int(ratio * park_info["capacity"])
    
    if DB:
        DB.collection("otoparklar").document(request.park_id).update({
            "current_occupancy_ratio": round(ratio, 2),
            "estimated_cars_now": cars,
            "last_updated": datetime.now()
        })

    return {"park_id": request.park_id, "occupancy_ratio": round(ratio, 2), "estimated_cars": cars}

# --- Endpoint 2: AKILLI ÖNERİ SİSTEMİ ---
@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    print(f"Yeni İstek Geldi! Hedef: {request.target_lat}, {request.target_lon}")
    results = []
    for park_id, info in PARK_MAP.items():
        # 1. ML Modeli ile Doluluk Tahmini
        occ_ratio = await calculate_occupancy_ratio(park_id, request.arrival_time)
        
        # 2. Google Distance Matrix (Mesafe ve Trafik)
        dist_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={info['latitude']},{info['longitude']}&destinations={request.target_lat},{request.target_lon}&departure_time=now&key={GOOGLE_MAPS_API_KEY}"
        dist_res = requests.get(dist_url).json()

        if park_id == "P_0": # Sadece ilk park için yazdır ki terminal dolmasın
             print(f"DEBUG Google Yanıtı: {dist_res}")
        
        try:
            element = dist_res['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                dist_km = element['distance']['value'] / 1000
                dur_min = (element['duration_in_traffic']['value'] if 'duration_in_traffic' in element else element['duration']['value']) / 60
            else:
                print(f"⚠️ Google API Durum Hatası ({park_id}): {element['status']}")
                dist_km, dur_min = 99.0, 99.0
        except Exception as e:
            print(f"❌ Veri Ayrıştırma Hatası: {e}")
            dist_km, dur_min = 99.0, 99.0

        # 3. Puanlama Algoritması (Düşük puan daha iyidir)
        score = (dist_km* 10) + (dur_min * 5) + (occ_ratio * 30)
        if occ_ratio > 0.85: score += 1000 # Doluysa cezalandır

        results.append({
            "park_id": park_id, "latitude": info["latitude"], "longitude": info["longitude"],
            "occupancy_ratio": round(occ_ratio, 2), "distance_km": round(dist_km, 2),
            "duration_min": round(dur_min, 1), "score": score
        })

    results.sort(key=lambda x: x['score'])
    return {"recommended_parking": results[0], # En yüksek skorlu olan
    "all_parkings": results}