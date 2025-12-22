import os
import requests
import pandas as pd
import numpy as np
import joblib
import firebase_admin
from datetime import datetime, timedelta
from typing import AsyncGenerator, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

model = None
scaler = None
DB = None
PARK_MAP = {}

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "firebase_admin_key.json")

# --- Yardımcı Fonksiyonlar ---
async def calculate_occupancy_ratio(park_id: str, target_time: datetime):
    park_info = PARK_MAP.get(park_id)
    if not park_info or model is None: return 0.5
    
    # ML model girdilerini hazırla (Basitleştirilmiş hava durumu varsayılanı ile)
    hour = target_time.hour
    dayofweek = target_time.weekday()
    
    df_input = pd.DataFrame([{
        "hour": hour, "dayofweek": dayofweek, "is_weekend": 1 if dayofweek >= 5 else 0,
        "is_holiday": 0, "park_id_encoded": park_info["encoded_id"],
        "max_capacity": park_info["capacity"], "temperature": 18.0,
        "precipitation": 0.0, "wind_speed": 5.0, "pressure": 1013.0
    }])

    scale_cols = ["hour", "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])
    
    FEATURES = ["hour", "dayofweek", "is_weekend", "is_holiday", "park_id_encoded", 
                "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]

    ratio = float(model.predict(df_input[FEATURES])[0])
    return max(0.0, min(1.0, ratio))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, DB, PARK_MAP
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
        DB = firestore.client()
        docs = DB.collection("otoparklar").stream()
        for doc in docs:
            data = doc.to_dict()
            PARK_MAP[data["park_id"]] = {
                "encoded_id": data.get("park_id_encoded"),
                "capacity": data.get("max_capacity"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            }
        model = joblib.load("retrained_occupancy_model.joblib")
        scaler = joblib.load("retrained_standard_scaler.joblib")
        print("✅ Sistem Hazır")
    except Exception as e: print(f"❌ Hata: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RecommendationRequest(BaseModel):
    target_lat: float
    target_lon: float
    user_lat: float # Flutter'dan gelen sanal konum
    user_lon: float

@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    results = []
    
    for park_id, info in PARK_MAP.items():
        # 1. Mevcut Konum -> Otopark (Sürüş Süresi + Trafik)
        # departure_time=now ekleyerek canlı trafik verisi alıyoruz
        dist_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                    f"origins={request.user_lat},{request.user_lon}&"
                    f"destinations={info['latitude']},{info['longitude']}&"
                    f"departure_time=now&" 
                    f"key={GOOGLE_MAPS_API_KEY}")
        
        try:
            dist_res = requests.get(dist_url).json()
            element = dist_res['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                dur_text = element.get('duration_in_traffic', element['duration'])['text']
                # 'duration_in_traffic' varsa onu kullan, yoksa normal duration
                dur_min = element.get('duration_in_traffic', element['duration'])['value'] / 60
                dist_km = element['distance']['value'] / 1000
                display_duration = dur_text.replace("mins", "").replace("min", "").strip()
            else:
                dur_min, display_duration = 120, "120" # Yol bulunamadıysa yüksek ceza
        except:
            dur_min, display_duration = 120, "120"

        # 2. Otopark -> Hedef (Yürüme Süresi)
        walk_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                    f"origins={info['latitude']},{info['longitude']}&"
                    f"destinations={request.target_lat},{request.target_lon}&"
                    f"mode=walking&"
                    f"key={GOOGLE_MAPS_API_KEY}")
        
        try:
            walk_res = requests.get(walk_url).json()
            w_element = walk_res['rows'][0]['elements'][0]
            if w_element['status'] == 'OK':
                display_walk = w_element['duration']['text'].replace("mins", "").replace("min", "").strip()
                walk_min = w_element['duration']['value'] / 60
            else:
                display_walk, walk_min = "120", 120
        except:
            display_walk, walk_min = "120", 120

        # 3. Varış Anındaki Doluluk Tahmini
        arrival_time = datetime.now() + timedelta(minutes=dur_min)
        occ_ratio = await calculate_occupancy_ratio(park_id, arrival_time)

        # 4. Akıllı Skorlama (Ağırlıkları dengeleyelim)
        # Yürüme süresi kullanıcı için daha kritiktir (Sürüşten daha yorucu)
        # Bu yüzden yürüme katsayısını 0.3'ten 0.5'e çıkarmanı öneririm.
        score = (dur_min * 0.35) + (walk_min * 0.2) + (occ_ratio * 100 * 0.45)
        
        # Kritik Doluluk Cezası (0.85 üstü risklidir)
        if occ_ratio > 0.85: 
            score += (occ_ratio * 200) 

        results.append({
            "park_id": park_id, 
            "latitude": info["latitude"], 
            "longitude": info["longitude"],
            "occupancy_ratio": round(occ_ratio, 2), 
            "distance_km": round(dist_km, 2),
            "duration_min": round(dur_min, 1), 
            "walk_min": round(walk_min, 1), 
            "score": score,
            "duration_min": display_duration, # Artık direkt Google'ın metin değeri gidiyor
            "walk_min": display_walk,         # Artık direkt Google'ın metin değeri gidiyor
            "raw_dur": dur_min,               # Skorlama için ham sayı lazım
            "raw_walk": walk_min,
        })

    # Skora göre sırala ve sonuçları dön
    results.sort(key=lambda x: x['score'])
    return {"recommended_parking": results[0], "all_parkings": results}