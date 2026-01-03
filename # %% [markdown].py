# %% [markdown]
# <a href="https://colab.research.google.com/github/haticeeakgull/Smart-Parking-Lot-System-/blob/main/Untitled3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import pandas as pd
import os
import time
# --- Google Drive Bağlantısı (Colab için Gerekli) ---
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from meteostat import Point, Daily, Hourly
from datetime import datetime ,timedelta
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error




# %%



# %%
drive.mount('/content/drive')

# 2. Düzeltilmiş Dosya Yolu
# Klasörünüzün ismini buraya tam olarak yazmalısınız (örneğin: 'LizbonParkVeri')
# Eğer ana dizinde (MyDrive) değilse, klasör adını doğru verin.
# Varsayalım ki klasör adınız 'ParkingData'
base_folder_name = 'parkingDataset' # <-- LÜTFEN KENDİ KLASÖR İSMİNİZİ YAZIN
base_path = os.path.join('/content/drive/MyDrive', base_folder_name)

file_names = [
    '1t2020.csv', # Varsayım: Ocak, Şubat, Mart verileri
    '2t2020.csv', # Varsayım: Nisan, Mayıs, Haziran verileri
    '4t2020.csv'  # Varsayım: Ekim, Kasım, Aralık verileri
]

all_data_frames = []

# --- Dosyaları Döngüyle Okuma ve Birleştirme ---
print("Veri setleri okunuyor ve birleştiriliyor...")
for name in file_names:
    file_path = os.path.join(base_path, name)
    try:
        # CSV'yi okurken formatı belirtmek faydalı olabilir
        df_temp = pd.read_csv(file_path,sep=";")
        all_data_frames.append(df_temp)
        print(f"'{name}' başarıyla yüklendi. Boyut: {df_temp.shape}")
    except FileNotFoundError:
        print(f"UYARI: '{name}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"'{name}' okunurken hata oluştu: {e}")

# Tüm veri setlerini tek bir DataFrame'de birleştirme
if all_data_frames:
    df_full = pd.concat(all_data_frames, ignore_index=True)
    print("\n--- BİRLEŞTİRİLMİŞ VERİ SETİNİN İLK 5 SATIRI ---")
    print(df_full.head())
    print(f"\nToplam Satır Sayısı: {len(df_full)}")
else:
    print("Hata: Hiçbir dosya başarıyla yüklenemedi. İşleme devam edilemez.")

# %%
print("\n--- Mevcut Sütun İsimleri ---")
print(df_full.columns.tolist())
df_full['datetime'] = pd.to_datetime(df_full['datetime'], format='%Y/%m/%d %H:%M', errors='coerce')
df_full = df_full.drop(columns=['position', 'entity_ts'])
df_full['hour'] = df_full['datetime'].dt.hour
df_full['dayofweek'] = df_full['datetime'].dt.dayofweek # Pazartesi=0, Pazar=6
df_full['is_weekend'] = (df_full['dayofweek'] >= 5).astype(int) # Hafta Sonu (Cmt/Pazar) = 1

print("\n--- 3. Adım Sonrası Veri Seti Bilgileri ---")
df_full.info()
print("\n--- Yeni Özellikler ve Son 5 Satır ---")
print(df_full[['datetime', 'hour', 'dayofweek', 'is_weekend', 'occupancy']].tail())

# Eksik (NaT) değer olup olmadığını kontrol edelim
missing_times = df_full['datetime'].isnull().sum()
if missing_times > 0:
    print(f"\n*** DİKKAT: {missing_times} adet geçersiz tarih/saat bulundu (NaT). Bu satırlar temizlenmelidir. ***")

# %%
df_clean = df_full.dropna(subset=['datetime'])

# 2. Temizlenmiş Veri Setini Kontrol Etme
print(f"\n--- Veri Temizleme Sonucu ---")
print(f"Orijinal Satır Sayısı: {len(df_full)}")
print(f"Temizlenmiş Satır Sayısı: {len(df_clean)}")
print(f"Kaldırılan Kayıt Sayısı: {len(df_full) - len(df_clean)}")

# %%
df = df_clean.copy()



le = LabelEncoder()
# 'id_park' sütunundaki string ifadeleri sayılara dönüştür
df['park_id_encoded'] = le.fit_transform(df['id_park'])

print("\n--- Son Kontrol (Temizlenmiş Veri) ---")
print(df[['id_park', 'park_id_encoded', 'datetime', 'hour', 'occupancy']].head())

# %%
portugal_holidays_2020 = [
    '2020/01/01',  # Yılbaşı
    '2020/04/10',  # Kutsal Cuma (Good Friday)
    '2020/04/13',  # Paskalya Pazartesi (Easter Monday)
    '2020/04/25',  # Özgürlük Günü (Freedom Day)
    '2020/05/01',  # İşçi Bayramı (Labour Day)
    '2020/06/10',  # Portekiz Günü (Portugal Day)
    '2020/08/15',  # Meryem Ana'nın Göğe Alınması (Assumption of Mary)
    '2020/10/05',  # Cumhuriyet Günü (Republic Day)
    '2020/12/01',  # Bağımsızlık Günü (Restoration of Independence)
    '2020/12/08',  # Kutsal Gebelik (Immaculate Conception)
    '2020/12/25'   # Noel (Christmas Day)
]
df['date_only'] = df['datetime'].dt.strftime('%Y/%m/%d')
df['is_holiday'] = df['date_only'].isin(portugal_holidays_2020).astype(int)
df = df.drop(columns=['date_only'])

# 3. Son Kontrol
print("\n--- Tatil Özelliği Kontrolü ---")
print(df[df['is_holiday'] == 1][['datetime', 'is_holiday', 'occupancy']].head()) # Tatil günlerinden birkaçı
print(df[['datetime', 'hour', 'is_weekend', 'is_holiday', 'occupancy']].head())

# %%



# %%
# 2. Lizbon Merkezi Koordinatlarını Belirle
LAT = 38.7223
LON = -9.1393
ALT = 110

# 3. Meteostat için Nokta Tanımlama
location = Point(LAT, LON, ALT)
start_date = df['datetime'].min().date()
end_date = df['datetime'].max().date()

print(f"Hava durumu verisi çekilecek tarih aralığı: {start_date} -> {end_date}")

start_dt = datetime.strptime(str(start_date), '%Y-%m-%d')
end_dt = datetime.strptime(str(end_date), '%Y-%m-%d')

# Saatlik veri çekme
data = Hourly(location, start_dt, end_dt)
data = data.fetch()

# Hava Durumu DataFrame'i
weather_df = data[['temp', 'prcp', 'wspd', 'pres']].copy()
weather_df.index.name = 'datetime'

print(f"\nÇekilen Saatlik Hava Durumu Kayıt Sayısı: {len(weather_df)}")

# 4. Yuvarlama
df['rounded_datetime'] = df['datetime'].dt.floor('H')
weather_df.index = weather_df.index.floor('H')

# 5. Hava Durumu Verilerini Ana DataFrame ile Birleştirme (SUFFIXES EKLENDİ)
# Sonek ekleyerek çakışan isimleri (temp, prcp vb.) ayırıyoruz.
# Meteostat'tan gelenleri '_wx' (Weather Extension) ile işaretleyelim.
df_merged = df.merge(weather_df,
                     left_on='rounded_datetime',
                     right_index=True,
                     how='left',
                     suffixes=('', '_wx')) # Orijinal sütunlara suffix eklenmez, yeni gelenlere '_wx' eklenir.

# Geçici ve gereksiz sütunları temizleme
df_merged = df_merged.drop(columns=['rounded_datetime'])

# Güncel DataFrame'imizi df olarak yeniden atayalım
df = df_merged.copy()

print("\n--- Hava Durumu Entegrasyonu Sonrası İlk 5 Satır Kontrolü ---")
# Yeni sütunlar temp_wx, prcp_wx vb. olarak gelmeli
print(df[['datetime', 'hour', 'is_holiday', 'temp_wx', 'prcp_wx', 'occupancy']].head())
print("\n--- Entegre Veri Seti Bilgileri (Yeni Sütunlar Görünmeli) ---")
df.info()

# %%
cols_to_keep = [
    'id_park', 'name', 'max_capacity', 'occupancy', 'datetime',
    'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
    'temp_wx', 'prcp_wx', 'wspd_wx', 'pres_wx'
]

current_cols = df.columns.tolist()
final_cols_to_keep = [col for col in cols_to_keep if col in current_cols]

# Sadece temizlenmiş sütunları tutan yeni bir DataFrame oluşturalım
df_clean = df[final_cols_to_keep].copy()

df_clean.rename(columns={
    'temp_wx': 'temperature',
    'prcp_wx': 'precipitation',
    'wspd_wx': 'wind_speed',
    'pres_wx': 'pressure'
}, inplace=True)

df_clean['occupancy_ratio'] = df_clean['occupancy'] / df_clean['max_capacity']


# 4. Eksik Hava Durumu Değerlerini Doldurma (Imputation)
# 'precipitation' (Yağış) sütununda çok sayıda <NA> vardı. Bunları 0.0 ile dolduralım,
# çünkü <NA> genellikle yağış olmadığı anlamına gelir. Diğerlerini ortalama ile dolduralım.
df_clean['precipitation'].fillna(0.0, inplace=True)

for col in ['temperature', 'wind_speed', 'pressure']:
    df_clean[col].fillna(df_clean[col].mean(), inplace=True)

df = df_clean.drop(columns=['id_park', 'name', 'occupancy'])

print("\n--- Nihai Temizlik Sonrası Veri Seti Kontrolü ---")
df.info()
print("\nNihai Veri Seti Başlığı:")
print(df.head())



# %%
# 'datetime' sütununu çıkarıyoruz, çünkü model eğitimi için sayısal özelliklere ihtiyacımız var.
# Ayrıca modelde kullanmayacağımız ve temizlenmemiş olabilecek eski sütunları da düşürelim.
cols_to_drop_final = ['datetime', 'occupancy', 'id_park', 'name']

# Sadece mevcut olan sütunları düşürmeyi deneyeceğiz.
current_columns = df.columns.tolist()
cols_to_safely_drop = [col for col in cols_to_drop_final if col in current_columns]

# Nihai DataFrame'i oluştur
df_model = df.drop(columns=cols_to_safely_drop, errors='ignore').copy()
# .copy() kullanmak FutureWarning'ı azaltmaya yardımcı olur.

# 1. Özellikler (X) ve Hedef (Y) Değişkenlerini Ayırma
TARGET = 'occupancy_ratio'
FEATURES = [
    'hour',
    'dayofweek',
    'is_weekend',
    'is_holiday',
    'park_id_encoded',
    'max_capacity',
    'temperature',
    'precipitation',
    'wind_speed',
    'pressure'
]

X = df_model[FEATURES]
Y = df_model[TARGET]

# Eksik kalan 1 satırı temizleyelim (occupancy_ratio'da 1 null vardı)
# Bu satırda hem X hem Y için aynı satırların düşürüldüğünden emin olmak için birlikte temizleyelim.
valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]

print(f"Eğitime Hazır Toplam Kayıt: {len(X)}")

# Buradan sonra Model Eğitim Koduna geçebilirsiniz
# ----------------------------------------------------------------------
# 2. Veri Setini Eğitim ve Test Setlerine Ayırma (Zaman Serisi)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]


# 3. Veri Ölçeklendirme (Standardizasyon)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cols_to_scale = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']
# Eğitim verisini ölçeklendir
X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
# Test verisini aynı scaler ile ölçeklendir
X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])


# 4. Modeli Oluşturma ve Eğitme (Random Forest Regressor)

print("\n--- Model Eğitimi Başlıyor (Random Forest) ---")
start_time = time.time()

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, Y_train)

end_time = time.time()
print(f"Eğitim Tamamlandı. Süre: {end_time - start_time:.2f} saniye.")


# 5. Tahmin Yapma ve Değerlendirme
Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)

print("\n--- Model Performans Değerlendirmesi (Ratio Tahmini) ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# 6. Özellik Önem Derecesi (Feature Importance)
feature_importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n--- Özellik Önem Derecesi ---")
print(feature_importances)

# %%
start_month = 7  # Temmuz
end_month = 9    # Eylül (Eylül dahil olduğu için 9'u dahil etmeliyiz)
year = 2020

# Tahmin için gerekli olan ilk ve son saatleri belirleyelim
# Temmuz'un ilk günü saat 00:00:00'dan başlayıp, Eylül'ün son günü saat 23:00:00'a kadar.
pred_start_date = datetime(year, start_month, 1, 0, 0, 0)
# Eylül 30'unun sonuna kadar veri çekmek için, Ekim 1'in başlangıcından hemen önce bitirelim.
pred_end_date = datetime(year, end_month, 30, 23, 0, 0) # Eylül 30'a kadar

# 2. Tahmin Edilecek Tarih/Saatleri Oluşturma (Park bazında)
# Benzersiz park kimliklerini alalım (eğitim setinde kullanılan sayısal kodlar)
unique_park_ids = df['park_id_encoded'].unique()

# Tahmin edilecek tüm zaman ve park kombinasyonlarını içeren boş DataFrame oluşturma
prediction_data = []
current_time = pred_start_date

# Her park için her saatlik veriyi oluştur
print("Tahmin edilecek zaman aralığı ve park kombinasyonları oluşturuluyor...")
while current_time <= pred_end_date:
    for park_id in unique_park_ids:
        prediction_data.append({
            'park_id_encoded': park_id,
            'datetime': current_time
        })
    current_time += timedelta(hours=1) # Bir sonraki saate geç

df_predict = pd.DataFrame(prediction_data)
print(f"Oluşturulan Tahmin Satır Sayısı: {len(df_predict)}")


# 3. Tahmin Edilecek Veriler İçin Zaman Özelliklerini Çıkarma
df_predict['hour'] = df_predict['datetime'].dt.hour
df_predict['dayofweek'] = df_predict['datetime'].dt.dayofweek
df_predict['is_weekend'] = (df_predict['dayofweek'] >= 5).astype(int)

# 4. Tatil Özelliklerini Ekleme
# Temmuz, Ağustos ve Eylül 2020 tatillerini kontrol etmeliyiz.
# Sizin orijinal tatil listeniz 2020'nin tamamını kapsayacak şekilde genişletilmelidir.
portugal_holidays_2020_full = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15',  # Yaz ayındaki tek resmi tatil (15 Ağustos)
    '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25'
]
df_predict['date_only'] = df_predict['datetime'].dt.strftime('%Y/%m/%d')
df_predict['is_holiday'] = df_predict['date_only'].isin(portugal_holidays_2020_full).astype(int)
df_predict = df_predict.drop(columns=['date_only'])

print("Tahmin Veri Seti Hazır (İlk 5 Satır):")
print(df_predict.head())

# %%
# --- Hava Durumu Verisini Çekme ve Entegrasyon (Tekrar) ---

# Bu kısımlar doğru çalışıyordu:
# [Meteostat veri çekme kısmı...]
# data_pred = Hourly(location, pred_start_date_weather, pred_end_date_weather).fetch()
# ...

# 1. Hava Durumu Verilerini Tahmin DataFrame'i ile Birleştirme
df_predict_merged = df_predict.merge(weather_df_pred,
                                     left_on='rounded_datetime',
                                     right_index=True,
                                     how='left',
                                     suffixes=('_pred', ''))
df_predict_merged = df_predict_merged.drop(columns=['rounded_datetime'])

# 2. Sütun İsimlerini Düzeltme
df_predict_merged.rename(columns={
    'temp': 'temperature',
    'prcp': 'precipitation',
    'wspd': 'wind_speed',
    'pres': 'pressure'
}, inplace=True)

# 3. max_capacity özelliğini tahmin setine ekleme (HATA DÜZELTİLDİ)
# Hata veren satırın düzeltilmiş hali:
# Amacımız: Her park_id_encoded için tek bir max_capacity değeri almak.
park_capacity_map = df.groupby('park_id_encoded')['max_capacity'].first().to_dict()

# Kapasiteyi yeni DataFrame'e eşleme
df_predict_merged['max_capacity'] = df_predict_merged['park_id_encoded'].map(park_capacity_map)

# Eksik kalırsa en sık kullanılan kapasite ile doldur (güvenlik için)
df_predict_merged['max_capacity'].fillna(df['max_capacity'].mode().iloc[0], inplace=True)
df_predict_merged['max_capacity'] = df_predict_merged['max_capacity'].astype(int) # Tam sayıya çevirme


# 4. Eksik Hava Durumu Değerlerini Doldurma (Imputation)
df_predict_merged['precipitation'].fillna(0.0, inplace=True)

for col in ['temperature', 'wind_speed', 'pressure']:
    df_predict_merged[col].fillna(df_predict_merged[col].mean(), inplace=True)


# 5. Özellikleri Ölçeklendirme (Eğitimde kullanılan scaler ile)
cols_to_scale_full = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']
X_new_scaled = df_predict_merged[cols_to_scale_full].copy()

# .loc kullanarak warning'i azaltma ve güvenli ölçekleme
X_new_scaled.loc[:, cols_to_scale_full] = scaler.transform(X_new_scaled[cols_to_scale_full])

# Geri kalan özellikleri birleştirip tahmin setini oluşturalım
X_new_other = df_predict_merged[['dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded']]

# X_new için sadece gerekli sütunları birleştirelim.
# Önemli: Concatenation'dan sonra, sütun sırasını modelin beklediği hale getireceğiz.
X_new_temp = pd.concat([X_new_other.reset_index(drop=True), X_new_scaled.reset_index(drop=True)], axis=1)

# Sütun sırasını modelin beklediği "FEATURES" listesiyle eşitleme
FEATURES = [
    'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
    'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
]
X_new = X_new_temp[FEATURES]


# 6. Tahmin Yapma
Y_predicted_ratio = model.predict(X_new)

# 7. Sonuçları DataFrame'e Ekleme
df_predict_merged['predicted_occupancy_ratio'] = Y_predicted_ratio
df_predict_merged['predicted_occupancy'] = (df_predict_merged['predicted_occupancy_ratio'] * df_predict_merged['max_capacity']).round().astype(int)


print("\n--- Tahmin Sonuçları (Temmuz Başı) ---")
print(df_predict_merged[df_predict_merged['datetime'].dt.month == 7][['datetime', 'park_id_encoded', 'max_capacity', 'temperature', 'predicted_occupancy_ratio', 'predicted_occupancy']].head(15))

# %%
ORIGINAL_FEATURES = [
    'datetime', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
    'park_id_encoded', 'max_capacity', 'temperature', 'precipitation',
    'wind_speed', 'pressure', 'occupancy_ratio'
]

# Orijinal (Temmuz-Eylül dışındaki) veriyi filtrele
# df, model eğitiminden önceki temizlenmiş orijinal DataFrame'inizdir.
df_clean_original = df[
    (df['datetime'].dt.month < 7) | (df['datetime'].dt.month > 9)
].copy()

# Sadece ihtiyacımız olan kolonları tutalım
df_clean_original = df_clean_original[ORIGINAL_FEATURES]


# 2. Tahmin Verisini Hazırlama
# df_predict_merged, tahmin sonuçlarının olduğu DataFrame'inizdir.
PREDICTED_FEATURES = [
    'datetime', 'park_id_encoded', 'max_capacity',
    'hour', 'dayofweek', 'is_weekend', 'is_holiday',
    'temperature', 'precipitation', 'wind_speed', 'pressure',
    'predicted_occupancy_ratio' # Tahmin edilen oran
]

df_predicted_final = df_predict_merged[PREDICTED_FEATURES].copy()

# Sütun adlarını orijinal formata çevirme
df_predicted_final.rename(columns={'predicted_occupancy_ratio': 'occupancy_ratio'}, inplace=True)


# 3. İki Veri Setini Birleştirme (Union)
final_data = pd.concat([df_clean_original, df_predicted_final], ignore_index=True)

# Veriyi tarih/zamana göre sıralama
final_data = final_data.sort_values(by=['datetime', 'park_id_encoded']).reset_index(drop=True)


print("\n--- Nihai Birleştirilmiş Veri Seti Kontrolü (TÜM ÖZELLİKLERLE) ---")
print(f"Toplam Satır Sayısı: {len(final_data)}")
print("Sütunlar:")
print(final_data.columns.tolist())
print("\nTemmuz Tahmin Başlangıcı:")
print(final_data[final_data['datetime'].dt.month == 7].head())

# %%
final_data.to_csv('2020_Park_Doluluk_Tahmin_Tamamlandi.csv', index=False)
print("Nihai veri seti başarıyla kaydedildi: 2020_Park_Doluluk_Tahmin_Tamamlandi.csv")
df_new=pd.read_csv('2020_Park_Doluluk_Tahmin_Tamamlandi.csv')
df_new.head()
#df_new.info()



# %%
from datetime import datetime, timedelta
import numpy as np

# --- Yardımcı Fonksiyonlar (Daha Önce Kullanılan Yapıları Tekrar Oluşturma) ---

# 1. Hava Durumu Çekme Fonksiyonu (Meteostat'ı kullanır)
def fetch_weather_for_prediction(dt_start, dt_end, location):
    """Belirtilen zaman aralığı için hava durumu verilerini çeker."""
    # Bu kısım için Meteostat kütüphanesinin kurulu ve API'nin erişilebilir olması gerekir.
    try:
        from meteostat import Hourly, Point
        data_pred = Hourly(location, dt_start, dt_end).fetch()
        weather_df_pred = data_pred[['temp', 'prcp', 'wspd', 'pres']].copy()
        weather_df_pred.index.name = 'datetime'
        weather_df_pred.index = weather_df_pred.index.floor('H')
        return weather_df_pred
    except Exception as e:
        print(f"Hava durumu çekilirken hata oluştu (Meteostat API erişimini kontrol edin): {e}")
        return None

# 2. Özellik Çıkarma ve Ölçekleme Fonksiyonu
def prepare_features_for_prediction(df_template, target_datetime, park_capacities_map, scaler_obj):
    """Tek bir zaman noktası için tüm gerekli özellikleri oluşturur, ölçekler ve X formatına getirir."""

    # Tüm park ID'leri için tek bir zaman dilimi oluştur
    current_park_ids = df_template['park_id_encoded'].unique()

    pred_data = []
    for park_id in current_park_ids:
        # Temel zaman özelliklerini hesapla
        hour = target_datetime.hour
        dayofweek = target_datetime.dayofweek
        is_weekend = (dayofweek >= 5).astype(int)

        # Tatil kontrolü (Basitleştirilmiş versiyon)
        # Gerçek uygulamada, tatil listesini kullanmalısınız. Burada sadece 2020 verisi olduğu için
        # sadece Ocak 1'i kontrol edelim veya dışarıdan bir tatil kontrol fonksiyonu çağıralım.
        # Güvenlik için: Tatil kontrolünü manuel yapmadan sadece ana özelliklere odaklanalım.
        is_holiday = 0 # Basitleştirme adına varsayılan 0

        # Kapasiteyi ekle
        capacity = park_capacities_map.get(park_id, df_template['max_capacity'].mode().iloc[0])

        pred_data.append({
            'hour': hour,
            'dayofweek': dayofweek,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'park_id_encoded': park_id,
            'max_capacity': capacity,
        })

    X_new = pd.DataFrame(pred_data)

    # Hava Durumu Çekimi (Varsayımsal olarak 19.0C ve 0.0 yağış alalım, çünkü gerçek çekimi burada yapamayız)
    # GERÇEKTE BURADA fetch_weather_for_prediction çağırmalısınız!
    X_new['temperature'] = 19.0  # Örnek bir değer
    X_new['precipitation'] = 0.0 # Örnek bir değer
    X_new['wind_speed'] = 5.0    # Örnek bir değer
    X_new['pressure'] = 1013.0   # Örnek bir değer

    # Ölçekleme
    cols_to_scale_full = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']

    # Özelliklerin sırasını ve isimlerini modelin beklediği gibi düzenle
    X_new_scaled = X_new[cols_to_scale_full].copy()
    X_new_scaled.loc[:, cols_to_scale_full] = scaler_obj.transform(X_new_scaled[cols_to_scale_full])

    # Nihai X formatını oluştur (Orijinal FEATURE sırasıyla)
    FEATURES = [
        'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
        'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
    ]

    X_final = pd.merge(X_new[['dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded', 'max_capacity']],
                       X_new_scaled,
                       on=['max_capacity'],
                       how='inner')

    # Sütun sırasını FEATURES listesine göre düzenle
    X_final = X_final[FEATURES]

    return X_final


# --- ANA ÖNERİ FONKSİYONU ---

def suggest_parking_and_predict(hours_ahead, df_base, model_obj, scaler_obj, park_capacities_map, location_point):
    """
    Belirtilen süre sonrası için doluluk tahmin eder ve en uygun parkı önerir.
    """

    # 1. Hedef Zamanı Belirleme
    # Tahmin anını (örneğin son veri noktası) alın
    last_known_time = df_base['datetime'].max()

    # Kullanıcının istediği ileri zamanı hesapla
    target_datetime = last_known_time + timedelta(hours=hours_ahead)
    print(f"\nTahmin Zamanı: {target_datetime} (Son bilinen zamandan {hours_ahead} saat sonra)")

    # 2. Özellikleri Hazırlama (Hava durumu çekimini burada yapmak gerekir)

    # --- GERÇEK UYGULAMADA YAPILMASI GEREKENLER (Hava Durumu Çekimi) ---
    # weather_data = fetch_weather_for_prediction(target_datetime, target_datetime + timedelta(hours=1), location_point)
    # Eğer weather_data boş gelirse, tahmin yapamayız.
    # -------------------------------------------------------------------

    # Şimdilik, sadece zaman/park özelliklerini hazırlıyoruz (Hava durumu elle varsayılmıştır)
    X_pred = prepare_features_for_prediction(df_base, target_datetime, park_capacities_map, scaler_obj)

    # 3. Tahmin Yapma
    Y_pred_ratio = model_obj.predict(X_pred)

    # Tahminleri DataFrame'e ekleme
    X_pred['predicted_occupancy_ratio'] = Y_pred_ratio

    # Doluluk sayısını hesaplama
    X_pred['predicted_occupancy'] = (X_pred['predicted_occupancy_ratio'] * X_pred['max_capacity']).round().astype(int)

    # 4. Öneri (Örneğin, EN AZ DOLU OLAN PARK)

    # En az doluluğa sahip parkı bul
    best_suggestion = X_pred.sort_values(by='predicted_occupancy', ascending=True).iloc[0]

    # Sonuçları formatlama
    suggestion_details = {
        "Tahmin Edilen Zaman": target_datetime.strftime('%Y-%m-%d %H:%M'),
        "Önerilen Park ID": best_suggestion['park_id_encoded'],
        "Tahmin Edilen Doluluk Oranı": f"{best_suggestion['predicted_occupancy_ratio']:.2f} ({best_suggestion['predicted_occupancy']} araç)",
        "Park Kapasitesi": best_suggestion['max_capacity'],
        "Tahmin Edilen Sıcaklık (Örnek)": X_pred.loc[X_pred['park_id_encoded'] == best_suggestion['park_id_encoded'], 'temperature'].iloc[0]
    }

    return suggestion_details, X_pred


# --- KULLANIM ÖRNEĞİ ---

# Varsayımlar (Bunları önceki adımlardan almalısınız)
# 1. Park Kapasiteleri Haritası (df'ten alınmalı)
park_capacities_map = df_new.groupby('park_id_encoded')['max_capacity'].first().to_dict()
# 2. Hava durumu için konumu tanımla (Örnek değerler)
LAT, LON, ALT = 38.7223, -9.1393, 50
location = Point(LAT, LON, ALT)


# KULLANICI GİRDİSİ: Yarım saat (0.5 saat) veya 1.5 saat sonrası için tahmin yapalım
hours_ahead_input = 1.5 # Kullanıcı 1.5 saat sonrasını sordu

# Tahmin ve Öneri Fonksiyonunu Çalıştırma
# NOT: Bu, hava durumu verisi çekme adımında hata verebilir, çünkü o kısım sadece simülasyon içerir.
suggestion, all_predictions = suggest_parking_and_predict(
    hours_ahead=hours_ahead_input,
    df_base=df, # Orijinal temizlenmiş/genişletilmiş df'i kullanıyoruz
    model_obj=model,
    scaler_obj=scaler,
    park_capacities_map=park_capacities_map,
    location_point=location
)

print("\n*** ÖNERİ SONUCU ***")
for key, value in suggestion.items():
    print(f"{key}: {value}")


