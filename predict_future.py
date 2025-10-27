import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import os

# --- 1. KAYDEDİLEN MODEL VE SCALER'I YÜKLEME ---
MODEL_FILENAME = 'retrained_occupancy_model.joblib'
SCALER_FILENAME = 'retrained_standard_scaler.joblib'

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    print(f"Model ve Scaler başarıyla yüklendi: {MODEL_FILENAME} & {SCALER_FILENAME}")
except FileNotFoundError:
    print(f"HATA: Model veya Scaler dosyası bulunamadı. Lütfen yeniden eğitim kodunu çalıştırıp bu dosyaları oluşturduğunuzdan emin olun.")
    exit()

FEATURES = [
    'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
    'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
]
TARGET = 'occupancy_ratio' # Bu yeni kodda kullanılmasa da tutulması iyidir.
cols_to_scale = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']

# Sabit Bilgileri (Park Haritası) Yükleme/Tanımlama
# NOT: Bu kısmı, ilk eğitim dosyasından (df_trainable) doğru veriyi çekerek/kaydederek otomatikleştirmelisin.
# Şimdilik örnek verilerle devam ediyoruz:
try:
    # Park ID'leri (0, 1, 2, 3...) ve kapasiteleri
    park_capacities_map = {0: 100, 1: 150, 2: 80, 3: 200} 
    park_id_mapping = {0: 'P_A', 1: 'P_B', 2: 'P_C', 3: 'P_D'} # Encoded ID'den Orijinal ID'ye
    UNIQUE_PARK_ENCODED_IDS = list(park_capacities_map.keys())

except Exception as e:
    print(f"UYARI: Park haritaları elle girildi. Lütfen bu haritaları kalıcı hale getirin. Hata: {e}")

# Konum ve Tatil Listesi
LAT, LON, ALT = 38.7223, -9.1393, 110 
location = Point(LAT, LON, ALT)

portugal_holidays_2020_full = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25']
# --- 2. SABİT BİLGİLERİ YÜKLEME (Park Kapasiteleri ve Kodlama) ---
# NOT: Bu bilgi ilk eğitim dosyasından (df_trainable) çıkarılmalıdır. 
# Hızlıca kodlama yapabilmek için, basitçe bir harita oluşturuyoruz.
# Gerçek projede, bu haritayı ve LabelEncoder'ı da kaydetmeniz gerekir.

# İlk eğitim dosyasından (df_trainable) park ID'lerini ve kapasitelerini almalıyız.
# Eğer df_trainable'ı kaydetmediyseniz, bu bilgiyi elde etmek için son eğitim dosyasını tekrar yüklemeniz gerekir:
try:
    # Sadece gerçek veriyi içeren dosyayı yükle
    TRAIN_DATA_PATH = '2020_Park_Doluluk_Tahmin_Temizlendi.csv' # Varsayımsal dosya adı, EN SON ÇALIŞAN EĞİTİM KISMINDAKİ df_trainable'ı temsil eden dosya
    
    # Hızlıca yeniden yükleyip haritaları çıkaralım (Bu, ilk eğitim adımında yapılması gereken bir kayıptır, ancak şimdi düzeltiyoruz)
    df_temp_for_maps = pd.read_csv('2020_Park_Doluluk_Tahmin_Temizlendi.csv') # Bu dosyanın var olduğunu varsayıyorum. Eğer yoksa, ilk betiğin temizlenmiş halini yüklemelisiniz.
    
    # Alternatif: İlk eğitim dosyasından park bilgilerini alın (Eğer o dosyanız elinizde yoksa, bu kısım manuel olarak girilmelidir.)
    # Bu örnekte, park bilgilerini (id_park, max_capacity) elle girmek zorunda kalabiliriz, 
    # çünkü sadece yeni eğitilmiş modeli yükledik.
    
    # PRATİK ÇÖZÜM: Geçici olarak, park ID'lerini ve kapasitelerini tahmin için gerekli olan temel yapıyı simüle edelim.
    # Gerçekte: park_capacities_map ve le (LabelEncoder) yüklenmeliydi.
    
    # Eğer park ID'leri sabit ise (1, 2, 3, 4):
    park_capacities_map = {0: 100, 1: 150, 2: 80, 3: 200} # Örnek: Park ID'leri 0'dan başlar (Encoded)
    park_id_mapping = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'} # Örnek: Encoded ID'den Orijinal ID'ye
    
    # Park ID'lerini kullanacağımız için encoded ID'leri FEATURE listesindeki ile eşleştirmeliyiz.
    UNIQUE_PARK_ENCODED_IDS = list(park_capacities_map.keys())

except Exception as e:
    print(f"UYARI: Park haritaları ve kodlaması yüklenemedi. Lütfen elle girilen örnek değerleri güncelleyin. Hata: {e}")
    # Devam etmek için varsayımsal değerler kullanılıyor.

# Konum Bilgileri (Örnek: Lizbon)
LAT, LON, ALT = 38.7223, -9.1393, 110 
location = Point(LAT, LON, ALT)

# Tatil Listesi (Tahmin dönemi için)
portugal_holidays_2020_full = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25'
]

# --- 3. TAHMİN ZAMAN ARALIĞINI BELİRLEME ---
# Örnek: 1 Ocak 2021 için 24 saatlik tahmin yapalım
PRED_YEAR = 2021
PRED_MONTH = 1
PRED_DAY = 1
HOURS_TO_PREDICT = 24 # Kaç saatlik tahmin yapacaksın?

pred_start_time = datetime(PRED_YEAR, PRED_MONTH, PRED_DAY, 0, 0, 0)
pred_end_time = pred_start_time + timedelta(hours=HOURS_TO_PREDICT - 1)

print(f"\nTahmin Başlangıcı: {pred_start_time}")
print(f"Tahmin Bitişi: {pred_end_time}")

# --- 4. TAHMİN İÇİN GEREKLİ VERİ ŞABLONUNU OLUŞTURMA ---
prediction_data = []
current_time = pred_start_time

while current_time <= pred_end_time:
    for park_id_encoded in UNIQUE_PARK_ENCODED_IDS:
        prediction_data.append({
            'datetime': current_time,
            'park_id_encoded': park_id_encoded
        })
    current_time += timedelta(hours=1)

df_predict = pd.DataFrame(prediction_data)
df_predict['hour'] = df_predict['datetime'].dt.hour
df_predict['dayofweek'] = df_predict['datetime'].dt.dayofweek
df_predict['is_weekend'] = (df_predict['dayofweek'] >= 5).astype(int)
df_predict['date_only'] = df_predict['datetime'].dt.strftime('%Y/%m/%d')
df_predict['is_holiday'] = df_predict['date_only'].isin(portugal_holidays_2020_full).astype(int)
df_predict = df_predict.drop(columns=['date_only'])

# Kapasite Ekleme
df_predict['max_capacity'] = df_predict['park_id_encoded'].map(park_capacities_map)
df_predict.dropna(subset=['max_capacity'], inplace=True) # Kapasitesi bilinmeyen parkları çıkar

# --- 5. GEREKLİ HAVA DURUMU VERİSİNİ ÇEKME ---
print(f"\nTahmin dönemi hava durumu verisi çekiliyor: {pred_start_time.date()} -> {pred_end_time.date()}")
weather_df_pred = Hourly(location, pred_start_time, pred_end_time).fetch()

if weather_df_pred.empty:
    print("HATA: Meteostat'tan hava durumu verisi çekilemedi. Lütfen internet bağlantınızı veya konum/tarih aralığınızı kontrol edin.")
    exit()

weather_df_pred.index = weather_df_pred.index.floor('H')

# Hava Durumu Entegrasyonu
df_predict_merged = df_predict.merge(
    weather_df_pred[['temp', 'prcp', 'wspd', 'pres']],
    left_on=df_predict['datetime'].dt.floor('H'),
    right_index=True,
    how='left',
    suffixes=('', '_wx')
)

# Sütunları Düzeltme ve Eksikleri Doldurma
df_predict_merged.rename(columns={'temp': 'temperature', 'prcp': 'precipitation', 'wspd': 'wind_speed', 'pres': 'pressure'}, inplace=True)

# Eksik Hava Durumu Doldurma (Basitçe ortalama ile veya 0 ile)
df_predict_merged['precipitation'].fillna(0.0, inplace=True)
for col in ['temperature', 'wind_speed', 'pressure']:
    df_predict_merged[col].fillna(df_predict_merged[col].mean(), inplace=True)


# --- 6. TAHMİN İÇİN ÖZELLİKLERİ HAZIRLAMA VE ÖLÇEKLENDİRME ---

X_new = df_predict_merged[FEATURES].copy()

# Ölçeklendirme (Sadece scaler.transform kullanılmalı!)
cols_to_scale = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']

# Özelliklerin sırasını ve türünü kontrol et
X_new.loc[:, cols_to_scale] = scaler.transform(X_new[cols_to_scale])

# --- 7. TAHMİN YAPMA ---
Y_pred_ratio = model.predict(X_new)
df_predict_merged['predicted_occupancy_ratio'] = Y_pred_ratio
df_predict_merged['predicted_occupancy'] = (df_predict_merged['predicted_occupancy_ratio'] * df_predict_merged['max_capacity']).round().astype(int)


# --- 8. SONUÇLARI GÖRSELLEŞTİRME VE KAYDETME ---

# Park ID'lerini orijinal ID'lere çevirelim (Eğer map varsa)
df_predict_merged['Original_Park_ID'] = df_predict_merged['park_id_encoded'].map(park_id_mapping).fillna('UNKNOWN')

final_prediction_output = df_predict_merged[[
    'datetime', 'Original_Park_ID', 'park_id_encoded', 'max_capacity',
    'temperature', 'predicted_occupancy_ratio', 'predicted_occupancy'
]].sort_values(by=['datetime', 'park_id_encoded'])

print("\n*** GELECEK TAHMİNİ BAŞARILI ***")
print(final_prediction_output.head(10)) # İlk 10 tahmini göster

# İstenirse CSV'ye kaydet
final_prediction_output.to_csv('gelecek_tahminler.csv', index=False)
