import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import os
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

# --- 1. KAYDEDİLEN DOSYALARI YÜKLEME ---
MODEL_FILENAME = 'retrained_occupancy_model.joblib'
SCALER_FILENAME = 'retrained_standard_scaler.joblib'
LABEL_ENCODER_FILENAME = 'park_label_encoder.joblib' # Eğitimde kaydettiğimiz encoder

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    # Eğer LabelEncoder kaydettiysen yükle, yoksa manuel map kullanmaya devam et
    if os.path.exists(LABEL_ENCODER_FILENAME):
        le = joblib.load(LABEL_ENCODER_FILENAME)
    print(f"Model ve Scaler başarıyla yüklendi.")
except FileNotFoundError:
    print(f"HATA: Model veya Scaler dosyası bulunamadı!")
    exit()

# 🔥 YENİ FEATURES LİSTESİ (minute eklendi)
FEATURES = [
    'hour', 'minute', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
    'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
]

# --- 2. PARK BİLGİLERİ ---
# Not: Eğitimde kullandığın park_id_encoded değerleri ve kapasiteler birebir aynı olmalı
park_capacities_map = {0: 100, 1: 150, 2: 80, 3: 200} # Kendi değerlerinle güncelle
UNIQUE_PARK_ENCODED_IDS = list(park_capacities_map.keys())

LAT, LON, ALT = 38.7223, -9.1393, 110 
location = Point(LAT, LON, ALT)

portugal_holidays_2020_full = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25'
]

# --- 3. TAHMİN ZAMAN ARALIĞI (30 DK HASSASİYET) ---
# Örneğin: Şu andan itibaren 5 saat boyunca her 30 dakikada bir tahmin yap
pred_start_time = datetime.now().replace(second=0, microsecond=0)
# En yakın 30 dakikaya yuvarla (opsiyonel ama daha temiz görünür)
pred_start_time = pred_start_time - timedelta(minutes=pred_start_time.minute % 30)

HOURS_TO_PREDICT = 5 
# Kaç adet 30 dakikalık periyot olacağını hesapla
intervals = HOURS_TO_PREDICT * 2 

# --- 4. VERİ ŞABLONUNU OLUŞTURMA ---
prediction_data = []
current_time = pred_start_time

for _ in range(intervals):
    for park_id_encoded in UNIQUE_PARK_ENCODED_IDS:
        prediction_data.append({
            'datetime': current_time,
            'park_id_encoded': park_id_encoded
        })
    # 🔥 ARTIK 30 DAKİKA EKLİYORUZ
    current_time += timedelta(minutes=30)

df_predict = pd.DataFrame(prediction_data)

# 🔥 YENİ ÖZELLİKLER
df_predict['hour'] = df_predict['datetime'].dt.hour
df_predict['minute'] = df_predict['datetime'].dt.minute # 🔥 Kritik ekleme
df_predict['dayofweek'] = df_predict['datetime'].dt.dayofweek
df_predict['is_weekend'] = (df_predict['dayofweek'] >= 5).astype(int)
df_predict['date_only'] = df_predict['datetime'].dt.strftime('%Y/%m/%d')
df_predict['is_holiday'] = df_predict['date_only'].isin(portugal_holidays_2020_full).astype(int)

# Kapasite Ekleme
df_predict['max_capacity'] = df_predict['park_id_encoded'].map(park_capacities_map)

# --- 5. HAVA DURUMU ---
print(f"Hava durumu çekiliyor...")
weather_df_pred = Hourly(location, pred_start_time, current_time).fetch()

# Hava durumu genelde saatliktir, 30. dakikalar için aynı saatin verisini kullanacağız
if not weather_df_pred.empty:
    weather_df_pred.index = weather_df_pred.index.floor('h') # 'H' -> 'h'
    df_predict_merged = df_predict.merge(
        weather_df_pred[['temp', 'prcp', 'wspd', 'pres']],
        left_on=df_predict['datetime'].dt.floor('h'),
        right_index=True,
        how='left'
    )
    df_predict_merged.rename(columns={'temp': 'temperature', 'prcp': 'precipitation', 
                                      'wspd': 'wind_speed', 'pres': 'pressure'}, inplace=True)
else:
    # Hava durumu çekilemezse varsayılan değerler
    for col in ['temperature', 'precipitation', 'wind_speed', 'pressure']:
        df_predict_merged[col] = 0.0

# Eksikleri doldur
df_predict_merged['precipitation'].fillna(0.0, inplace=True)
for col in ['temperature', 'wind_speed', 'pressure']:
    df_predict_merged[col].fillna(df_predict_merged[col].mean() if not df_predict_merged[col].isnull().all() else 0.0, inplace=True)

# --- 6. ÖLÇEKLENDİRME ---
# Hatanın çözümü: Scaler'a tüm sütunları (FEATURES listesindeki sırayla) gönderiyoruz
X_new = df_predict_merged[FEATURES].copy()

# Scaler, eğitim sırasında gördüğü TÜM sütun isimlerini bekler.
# X_new zaten FEATURES listesindeki sıraya göre oluşturuldu.
X_new_scaled_values = scaler.transform(X_new)

# Dönüştürülmüş değerleri içeren yeni bir DataFrame oluşturuyoruz (Sütun isimlerini korumak için)
X_new_final = pd.DataFrame(X_new_scaled_values, columns=FEATURES)

# --- 7. TAHMİN ---
# Artık modelin beklediği tam ölçeklenmiş veriyi gönderiyoruz
Y_pred_ratio = model.predict(X_new_final)

df_predict_merged['predicted_occupancy_ratio'] = Y_pred_ratio
df_predict_merged['predicted_occupancy'] = (df_predict_merged['predicted_occupancy_ratio'] * df_predict_merged['max_capacity']).round().astype(int)
# --- 8. SONUÇ ---
final_output = df_predict_merged[['datetime', 'park_id_encoded', 'predicted_occupancy']]
print("\n--- 30 Dakikalık Tahminler Başarıyla Üretildi ---")
print(final_output.head(10))

final_output.to_csv('gelecek_tahminler_30dk.csv', index=False)