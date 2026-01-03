import os
import requests
import pandas as pd
import joblib
import firebase_admin
import numpy as np
import warnings
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Uyarıları ve Pandas uyarısını gizle
warnings.filterwarnings('ignore')
load_dotenv()

# Global Değişkenler
model = None
scaler = None
DB = None
PARK_MAP = {}

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "firebase_admin_key.json")

# --- Pydantic Modelleri ---
class RecommendationRequest(BaseModel):
    target_lat: float
    target_lon: float
    user_lat: float
    user_lon: float
    max_walk_time: int 

class PredictionRequest(BaseModel):
    park_id: str
    prediction_time: str

# --- Yardımcı Fonksiyonlar ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """İki nokta arasındaki kuş uçuşu mesafeyi (km) hesaplar."""
    if None in [lat1, lon1, lat2, lon2]: return 999.0
    R = 6371
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    return R * 2 * asin(sqrt(a))

async def calculate_occupancy_ratio(park_id: str, target_time: datetime):
    """Modeli kullanarak 30 dakikalık hassasiyetle doluluk tahmini yapar."""
    park_info = PARK_MAP.get(park_id)
    if not park_info or model is None or scaler is None: 
        return 0.5 
    
    try:
        hour = target_time.hour
        minute = target_time.minute
        dayofweek = target_time.weekday()
        is_weekend = 1 if dayofweek >= 5 else 0
        
        park_id_encoded = int(park_info.get("encoded_id", 0))
        capacity = int(park_info.get("capacity", 50))

        # 🔥 KRİTİK: Eğitimdeki FEATURES sırası tam olarak bu olmalı
        FEATURES = [
            "hour", "minute", "dayofweek", "is_weekend", "is_holiday", 
            "park_id_encoded", "max_capacity", "temperature", "precipitation", 
            "wind_speed", "pressure"
        ]
        
        input_data = {
            "hour": hour,
            "minute": minute,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_holiday": 0,
            "park_id_encoded": park_id_encoded,
            "max_capacity": capacity,
            "temperature": 20.0, 
            "precipitation": 0.0,
            "wind_speed": 5.0,
            "pressure": 1013.0
        }
        
        # DataFrame oluştur ve sütun sırasını FEATURES listesine göre sabitle
        df_input = pd.DataFrame([input_data])[FEATURES]
        
        # 🔥 HATA ÇÖZÜMÜ: Scaler'a sadece scale_cols değil, TÜM sütunları gönderiyoruz.
        # Çünkü scaler fit edilirken tüm özellikleri gördü.
        df_input_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURES)

        # Tahmin
        prediction = model.predict(df_input_scaled)
        base_ratio = float(prediction[0])

        # Yoğunluk saatlerine göre küçük düzeltme
        time_factor = 0.0
        if 8 <= hour <= 10: time_factor = 0.05
        elif 17 <= hour <= 19: time_factor = 0.10

        final_ratio = max(0.05, min(0.98, base_ratio + time_factor))
        return final_ratio

    except Exception as e:
        print(f"❌ Tahmin hatası ({park_id}): {e}")
        return 0.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken yeni 30dk modellerini yükler."""
    global model, scaler, DB, PARK_MAP
    try:
        if not firebase_admin._apps:
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
        
        # 🔥 Yeni 30 dk'lık dosyaları yükle
        model = joblib.load("retrained_occupancy_model.joblib")
        scaler = joblib.load("retrained_standard_scaler.joblib")
        print(f"✅ 30 DK Modeli Hazır: {len(PARK_MAP)} otopark yüklendi.")
    except Exception as e: 
        print(f"❌ Başlatma Hatası: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):
    """Tekil otopark için 30 dk hassasiyetli tahmin."""
    try:
        clean_time = request.prediction_time.replace("Z", "").split(".")[0]
        target_time = datetime.fromisoformat(clean_time)
        ratio = await calculate_occupancy_ratio(request.park_id, target_time)
        return {"park_id": request.park_id, "predicted_occupancy_ratio": round(ratio, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    """Öneride de 30 dk hassasiyetli varış tahmini kullanılır."""
    MAX_RADIUS_KM = 2.0 
    filtered_parks = []
    for p_id, info in PARK_MAP.items():
        if info.get('latitude') is None: continue
        air_dist = haversine_distance(request.target_lat, request.target_lon, info['latitude'], info['longitude'])
        if air_dist <= MAX_RADIUS_KM: 
            filtered_parks.append((p_id, info))

    if not filtered_parks:
        for p_id, info in PARK_MAP.items():
            air_dist = haversine_distance(request.target_lat, request.target_lon, info['latitude'], info['longitude'])
            if air_dist <= 5.0: filtered_parks.append((p_id, info))

    if not filtered_parks:
        raise HTTPException(status_code=404, detail="Yakınlarda otopark bulunamadı.")

    filtered_parks.sort(key=lambda x: haversine_distance(request.target_lat, request.target_lon, x[1]['latitude'], x[1]['longitude']))
    limited_parks = filtered_parks[:5] 
    dest_str = "|".join([f"{p[1]['latitude']},{p[1]['longitude']}" for p in limited_parks])
    
    dist_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                f"origins={request.user_lat},{request.user_lon}&"
                f"destinations={dest_str}&mode=driving&key={GOOGLE_MAPS_API_KEY}")
    
    walk_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                f"origins={dest_str}&"
                f"destinations={request.target_lat},{request.target_lon}&"
                f"mode=walking&key={GOOGLE_MAPS_API_KEY}")

    try:
        d_res = requests.get(dist_url).json()
        w_res = requests.get(walk_url).json()

        if d_res.get('status') != 'OK' or w_res.get('status') != 'OK':
            raise HTTPException(status_code=502, detail="Google API Hatası")

        results = []
        for i, (park_id, info) in enumerate(limited_parks):
            try:
                d_elem = d_res['rows'][0]['elements'][i]
                w_elem = w_res['rows'][i]['elements'][0]

                if d_elem['status'] == 'OK' and w_elem['status'] == 'OK':
                    dur_min = d_elem['duration']['value'] / 60
                    walk_min = w_elem['duration']['value'] / 60
                    
                    arrival_time = datetime.now() + timedelta(minutes=dur_min)
                    # Buradaki tahmin artık 30 dk hassasiyetli
                    occ_ratio = await calculate_occupancy_ratio(park_id, arrival_time)

                    score = (walk_min * 3.0) + (dur_min * 0.8) + (occ_ratio * 20)
                    if walk_min > request.max_walk_time:
                        score += (walk_min - request.max_walk_time) * 40 

                    results.append({
                        "park_id": park_id,
                        "latitude": info["latitude"],
                        "longitude": info["longitude"],
                        "occupancy_ratio": round(occ_ratio, 2),
                        "duration_min": round(dur_min),
                        "walk_min": round(walk_min),
                        "score": score
                    })
            except: continue

        results.sort(key=lambda x: x['score'])
        return {"recommended_parking": results[0], "all_parkings": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)