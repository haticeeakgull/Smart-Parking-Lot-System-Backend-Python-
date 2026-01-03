import os
import requests
import pandas as pd
import joblib
import firebase_admin
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri yükle
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
    """Modeli kullanarak doluluk tahmini yapar."""
    park_info = PARK_MAP.get(park_id)
    if not park_info or model is None or scaler is None: 
        return 0.5 # Bilgi yoksa %50 döndür (Sistem çökmesin)
    
    try:
        hour = target_time.hour
        dayofweek = target_time.weekday()
        is_weekend = 1 if dayofweek >= 5 else 0
        
        # Veri tipi dönüşümleri (Firestore güvenliği)
        park_id_encoded = int(park_info.get("encoded_id", 0))
        capacity = int(park_info.get("capacity", 50))

        # Model girdisi
        input_data = {
            "hour": hour,
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
        
        df_input = pd.DataFrame([input_data])
        scale_cols = ["hour", "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]
        
        # Ölçeklendirme
        df_input[scale_cols] = scaler.transform(df_input[scale_cols])

        # Tahmin Özellikleri
        FEATURES = ["hour", "dayofweek", "is_weekend", "is_holiday", "park_id_encoded", 
                    "max_capacity", "temperature", "precipitation", "wind_speed", "pressure"]
        
        prediction = model.predict(df_input[FEATURES])
        base_ratio = float(prediction[0])

        # 🚀 ZAMANSAL SALINIM (Kullanıcı etkileşimi için)
        time_factor = 0.0
        if 8 <= hour <= 10: time_factor = 0.07   # Sabah yoğunluğu
        elif 12 <= hour <= 14: time_factor = 0.12 # Öğle yoğunluğu
        elif 17 <= hour <= 19: time_factor = 0.18 # Akşam zirve
        elif 22 <= hour <= 23 or 0 <= hour <= 6: time_factor = -0.25 # Gece boşalması

        final_ratio = max(0.05, min(0.98, base_ratio + time_factor))
        return final_ratio
    except Exception as e:
        print(f"❌ Tahmin hatası ({park_id}): {e}")
        return 0.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken modelleri ve veritabanını yükler."""
    global model, scaler, DB, PARK_MAP
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
        
        DB = firestore.client()
        # Firestore'daki otoparkları belleğe al (Hız için)
        docs = DB.collection("otoparklar").stream()
        for doc in docs:
            data = doc.to_dict()
            PARK_MAP[data["park_id"]] = {
                "encoded_id": data.get("park_id_encoded"),
                "capacity": data.get("max_capacity"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            }
        
        # Model ve Scaler dosyalarını yükle
        model = joblib.load("retrained_occupancy_model.joblib")
        scaler = joblib.load("retrained_standard_scaler.joblib")
        print(f"✅ Sistem Hazır: {len(PARK_MAP)} otopark yüklendi.")
    except Exception as e: 
        print(f"❌ Başlatma Hatası: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):
    """Tekil bir otopark için tahmin döndürür."""
    try:
        clean_time = request.prediction_time.replace("Z", "").split(".")[0]
        target_time = datetime.fromisoformat(clean_time)
        ratio = await calculate_occupancy_ratio(request.park_id, target_time)
        return {"park_id": request.park_id, "predicted_occupancy_ratio": round(ratio, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    """Fatura dostu akıllı otopark önerisi yapar (Debug Modu Aktif)."""
    
    # 1. ADIM: KUŞ UÇUŞU FİLTRELEME (Bedava)
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

    # 2. ADIM: EN YAKIN 5 TANEYİ SEÇ
    filtered_parks.sort(key=lambda x: haversine_distance(request.target_lat, request.target_lon, x[1]['latitude'], x[1]['longitude']))
    limited_parks = filtered_parks[:5] 

    dest_str = "|".join([f"{p[1]['latitude']},{p[1]['longitude']}" for p in limited_parks])
    
    # URL Oluşturma
    dist_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                f"origins={request.user_lat},{request.user_lon}&"
                f"destinations={dest_str}&mode=driving&key={GOOGLE_MAPS_API_KEY}")
    
    walk_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                f"origins={dest_str}&"
                f"destinations={request.target_lat},{request.target_lon}&"
                f"mode=walking&key={GOOGLE_MAPS_API_KEY}")

    # 🔍 DEBUG: İSTEKLERİ TERMİNALE BAS
    print("\n--- GOOGLE API DEBUG BAŞLADI ---")
    print(f"📡 Driving API URL: {dist_url}")
    print(f"📡 Walking API URL: {walk_url}")

    try:
        d_res_raw = requests.get(dist_url)
        w_res_raw = requests.get(walk_url)
        
        d_res = d_res_raw.json()
        w_res = w_res_raw.json()

        # 🔍 DEBUG: GOOGLE YANITLARINI GÖR
        print(f"📦 Driving Response Status: {d_res.get('status')}")
        if d_res.get('status') != 'OK':
            print(f"❌ Driving Error Detail: {d_res.get('error_message', 'No extra message')}")

        print(f"📦 Walking Response Status: {w_res.get('status')}")
        if w_res.get('status') != 'OK':
            print(f"❌ Walking Error Detail: {w_res.get('error_message', 'No extra message')}")
        print("--- DEBUG BİTTİ ---\n")

        if d_res.get('status') != 'OK' or w_res.get('status') != 'OK':
            google_error = d_res.get('error_message') or w_res.get('error_message') or "Bilinmeyen API Hatası"
            raise HTTPException(status_code=502, detail=f"Google API Hatası: {google_error}")

        results = []
        for i, (park_id, info) in enumerate(limited_parks):
            try:
                d_elem = d_res['rows'][0]['elements'][i]
                w_elem = w_res['rows'][i]['elements'][0]

                if d_elem['status'] == 'OK' and w_elem['status'] == 'OK':
                    dur_min = d_elem['duration']['value'] / 60
                    walk_min = w_elem['duration']['value'] / 60
                    
                    arrival_time = datetime.now() + timedelta(minutes=dur_min)
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
            except Exception as e: 
                print(f"⚠️ Element ayrıştırma hatası: {e}")
                continue

        if not results:
            raise HTTPException(status_code=404, detail="Uygulun rota bulunamadı.")

        results.sort(key=lambda x: x['score'])
        return {"recommended_parking": results[0], "all_parkings": results}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🔥 Kritik Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)