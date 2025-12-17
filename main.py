import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# --- Global Değişkenler ---
model = None
scaler = None
FIREBASE_APP = None
DB = None
PARK_MAP = {}

HOLIDAYS = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25',
]

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model, scaler, FIREBASE_APP, DB, PARK_MAP

    try:
        # Firebase
        cred = credentials.Certificate("firebase_admin_key.json")
        FIREBASE_APP = firebase_admin.initialize_app(cred)
        DB = firestore.client()
        print("✅ Firebase bağlantısı kuruldu")

        # Otopark verileri
        docs = DB.collection("otoparklar").stream()
        for doc in docs:
            data = doc.to_dict()
            park_id = data.get("park_id")
            encoded_id = data.get("park_id_encoded")
            capacity = data.get("max_capacity")

            if park_id and encoded_id is not None and capacity is not None:
                PARK_MAP[park_id] = {
                    "encoded_id": encoded_id,
                    "capacity": capacity,
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                }

        print(f" {len(PARK_MAP)} otopark yüklendi")

        # Model & scaler
        model = joblib.load("retrained_occupancy_model.joblib")
        scaler = joblib.load("retrained_standard_scaler.joblib")
        print(" Model ve Scaler yüklendi")

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    yield
    print(" Uygulama kapanıyor")


app = FastAPI(title="Otopark Doluluk Tahmin API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklardan gelen isteklere izin ver (Geliştirme için güvenli)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Hava Durumu ---
def get_weather_forecast(target_time: datetime, lat=38.7223, lon=-9.1393):
    cache = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
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

        target_time = (
            pd.to_datetime(target_time).tz_localize("UTC")
            if target_time.tzinfo is None else target_time
        )

        row = df.iloc[(df["date"] - target_time).abs().argsort()[:1]]

        return {
            "temperature": float(row["temperature"].values[0]),
            "precipitation": float(row["precipitation"].values[0]),
            "wind_speed": float(row["wind_speed"].values[0]),
            "pressure": float(row["pressure"].values[0]),
        }

    except Exception:
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 10.0, "pressure": 1013.0}


class PredictionRequest(BaseModel):
    park_id: str
    prediction_time: datetime


@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil")

    park_info = PARK_MAP.get(request.park_id)
    if not park_info:
        raise HTTPException(status_code=404, detail="Geçersiz park_id")

    dt = request.prediction_time
    hour = dt.hour
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    is_holiday = 1 if dt.strftime("%Y/%m/%d") in HOLIDAYS else 0

    weather = get_weather_forecast(
        dt,
        lat=park_info.get("latitude", 38.7223),
        lon=park_info.get("longitude", -9.1393),
    )

    df = pd.DataFrame([{
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "park_id_encoded": park_info["encoded_id"],
        "max_capacity": park_info["capacity"],
        "temperature": weather["temperature"],
        "precipitation": weather["precipitation"],
        "wind_speed": weather["wind_speed"],
        "pressure": weather["pressure"],
    }])

    scale_cols = ["hour", "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    FEATURES = [
        "hour", "dayofweek", "is_weekend", "is_holiday",
        "park_id_encoded", "max_capacity",
        "temperature", "precipitation", "wind_speed", "pressure"
    ]

    ratio = float(model.predict(df[FEATURES])[0])
    ratio = max(0.0, min(1.0, ratio))
    cars = int(ratio * park_info["capacity"])

    color = "GREEN"
    if ratio > 0.25: color = "YELLOW"
    if ratio > 0.5: color = "ORANGE"
    if ratio > 0.85: color = "RED"

    if DB:
        DB.collection("otoparklar").document(request.park_id).update({
            "current_occupancy_ratio": round(ratio, 2),
            "estimated_cars_now": cars,
            "last_updated": datetime.now()
        })

    return {
        "park_id": request.park_id,
        "prediction_time": dt,
        "occupancy_ratio": round(ratio, 2),
        "estimated_cars": cars,
        "max_capacity": park_info["capacity"],
        "status": color,
        "weather_summary": weather
    }
