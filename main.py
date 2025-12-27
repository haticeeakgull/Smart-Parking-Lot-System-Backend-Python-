import os
import requests
import pandas as pd
import joblib
import firebase_admin
from datetime import datetime
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

# --- Modeller ---
class RecommendationRequest(BaseModel):
    target_lat: float
    target_lon: float
    user_lat: float
    user_lon: float
    max_walk_time: int  # Flutter'dan gelen zorunlu parametre

class PredictionRequest(BaseModel):
    park_id: str
    prediction_time: str

# --- Yardımcı Fonksiyonlar ---
async def calculate_occupancy_ratio(park_id: str, target_time: datetime):
    park_info = PARK_MAP.get(park_id)
    if not park_info or model is None: return 0.5
    
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
    #test kodu
    # if target_time.hour >= 18:
    #      return 0.99

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
    except Exception as e: print(f"❌ Başlatma Hatası: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):
    try:
        # Hatalı olan 'fromiso8601' yerine 'fromisoformat' kullanıyoruz
        clean_time = request.prediction_time.replace("Z", "")
        target_time = datetime.fromisoformat(clean_time)
        
        ratio = await calculate_occupancy_ratio(request.park_id, target_time)
        return {
            "park_id": request.park_id, 
            "predicted_occupancy_ratio": round(ratio, 2)
        }
    except Exception as e:
        # Hatanın ne olduğunu terminalde görmek için print kalsın
        print(f"❌ Tahmin Hatası Detayı: {e}")
        raise HTTPException(status_code=400, detail=f"Geçersiz tarih formatı: {str(e)}")

@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    results = []
    
    for park_id, info in PARK_MAP.items():
        # Sürüş Süresi (Kullanıcıdan Otoparka)
        dist_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                    f"origins={request.user_lat},{request.user_lon}&"
                    f"destinations={info['latitude']},{info['longitude']}&"
                    f"departure_time=now&key={GOOGLE_MAPS_API_KEY}")
        
        # Yürüme Süresi (Otoparktan Hedefe)
        walk_url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
                    f"origins={info['latitude']},{info['longitude']}&"
                    f"destinations={request.target_lat},{request.target_lon}&"
                    f"mode=walking&key={GOOGLE_MAPS_API_KEY}")

        try:
            d_res = requests.get(dist_url).json()
            w_res = requests.get(walk_url).json()
            
            if d_res['rows'][0]['elements'][0]['status'] == 'OK' and w_res['rows'][0]['elements'][0]['status'] == 'OK':
                dur_min = d_res['rows'][0]['elements'][0]['duration']['value'] / 60
                walk_min = w_res['rows'][0]['elements'][0]['duration']['value'] / 60
                
                occ_ratio = await calculate_occupancy_ratio(park_id, datetime.now())
                
                # --- AKILLI SKORLAMA (Yürüme Odaklı) ---
                # Düşük skor = Daha iyi seçim
                score = (walk_min * 2.5) + (dur_min * 0.5) + (occ_ratio * 15)
                
                # Eğer yürüme süresi sınırı aşıyorsa her aşan dakika için ceza puanı ekle
                if walk_min > request.max_walk_time:
                    score += (walk_min - request.max_walk_time) * 30 

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
            print(f"Hata {park_id}: {e}")
            continue

    if not results:
        raise HTTPException(status_code=404, detail="Uygun otopark verisi bulunamadı.")
    
    # Skora göre küçükten büyüğe sırala
    results.sort(key=lambda x: x['score'])
    
    return {"recommended_parking": results[0], "all_parkings": results}