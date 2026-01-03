import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime
from meteostat import Point, Hourly
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Uyarıları gizle
warnings.filterwarnings('ignore')

# --- 1. VERİ YÜKLEME ---
file_name = "2020_Park_Doluluk_Tahmin_Tamamlandi (1).csv"
print("Veri seti okunuyor...")

if os.path.exists(file_name):
    df = pd.read_csv(file_name, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    print(f"'{file_name}' yüklendi. Sütunlar: {df.columns.tolist()}")
else:
    print(f"HATA: {file_name} bulunamadı!")
    exit()

# --- 2. ZAMAN ÖZELLİKLERİ VE 30 DK YUVARLAMA ---
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])

# 30 dakikalık hassasiyet
df['rounded_datetime'] = df['datetime'].dt.floor('30min')
df['hour'] = df['rounded_datetime'].dt.hour
df['minute'] = df['rounded_datetime'].dt.minute
df['dayofweek'] = df['rounded_datetime'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Park ID Kodlama
le = LabelEncoder()
if 'id_park' in df.columns:
    df['park_id_encoded'] = le.fit_transform(df['id_park'])
elif 'park_id_encoded' not in df.columns:
    df['park_id_encoded'] = 0

# Tatil Listesi
portugal_holidays_2020 = ['2020/01/01', '2020/08/15', '2020/12/25'] # Örnek kısa liste
df['is_holiday'] = df['rounded_datetime'].dt.strftime('%Y/%m/%d').isin(portugal_holidays_2020).astype(int)

# --- 3. HAVA DURUMU (HATA KORUMALI) ---
# Eğer halihazırda sütunlar varsa dokunma, yoksa çekmeyi dene
weather_cols = ['temperature', 'precipitation', 'wind_speed', 'pressure']
for col in weather_cols:
    if col not in df.columns:
        df[col] = 0.0 # Varsayılan değer

print("Hava durumu kontrolü ve temizliği yapılıyor...")
# Basit ve güvenli doldurma:
df['temperature'] = pd.to_numeric(df.get('temperature'), errors='coerce').fillna(20.0)
df['precipitation'] = pd.to_numeric(df.get('precipitation'), errors='coerce').fillna(0.0)
df['wind_speed'] = pd.to_numeric(df.get('wind_speed'), errors='coerce').fillna(5.0)
df['pressure'] = pd.to_numeric(df.get('pressure'), errors='coerce').fillna(1013.0)

# Doluluk Oranı
if 'occupancy_ratio' not in df.columns:
    df['occupancy_ratio'] = df['occupancy'] / df['max_capacity']

df = df.dropna(subset=['occupancy_ratio'])

# --- 4. MODEL EĞİTİMİ ---
FEATURES = ['hour', 'minute', 'dayofweek', 'is_weekend', 'is_holiday', 
            'park_id_encoded', 'max_capacity', 'temperature', 'precipitation', 
            'wind_speed', 'pressure']
TARGET = 'occupancy_ratio'

X = df[FEATURES]
y = df[TARGET]

# Eğitim/Test ayırımı
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n🚀 Model Eğitiliyor (30 dk Hassasiyetle - Kayıt Sayısı: {len(X)})...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 5. KAYDETME ---
joblib.dump(model, 'retrained_occupancy_model.joblib')
joblib.dump(scaler, 'retrained_standard_scaler.joblib')
joblib.dump(le, 'park_label_encoder.joblib')

print("\n✅ BAŞARILI! Dosyalar oluşturuldu:")
print("- retrained_occupancy_model.joblib")
print("- retrained_standard_scaler.joblib")
print("- park_label_encoder.joblib")