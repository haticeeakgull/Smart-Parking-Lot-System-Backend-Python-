import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
import joblib 
import time

FILE_PATH = '2020_Park_Doluluk_Tahmin_Tamamlandi (1).csv'

# 1. Dosyayı Yükle
try:
    df_final = pd.read_csv(FILE_PATH)
    print(f"'{FILE_PATH}' başarıyla yüklendi. Toplam Kayıt: {len(df_final)}")
except FileNotFoundError:
    print(f"HATA: '{FILE_PATH}' dosyası bulunamadı. Lütfen ilk betiği çalıştırdığınızdan emin olun.")
    exit()

# 2. 'datetime' sütununu tekrar datetime tipine çevir
df_final['datetime'] = pd.to_datetime(df_final['datetime'])

# Filtrelenecek aylar (Temmuz=7, Ağustos=8, Eylül=9)
months_to_exclude = [7, 8, 9]

# Sadece bu aylara AİT OLMAYAN satırları seçiyoruz (Gerçek Gözlemlenen Veri)
df_trainable = df_final[
    ~df_final['datetime'].dt.month.isin(months_to_exclude)
].copy()

# Kesin Temizlik Adımı: X ve Y'de kalmış olabilecek tüm NaN değerlerini temizle
FEATURES = [
    'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
    'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
]
TARGET = 'occupancy_ratio'

df_trainable.dropna(subset=FEATURES + [TARGET], inplace=True) 
print(f"Temizlik sonrası Model eğitimi için kullanılacak GERÇEK Kayıt Sayısı: {len(df_trainable)}")

X = df_trainable[FEATURES]
Y = df_trainable[TARGET]

# Veri setinin son %20'sini test için ayırın (Zaman serisi düzenini koruyarak)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

print(f"Eğitim Seti: {len(X_train)}, Test Seti: {len(X_test)}")
scaler = StandardScaler()
cols_to_scale = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']

# Scaler'ı sadece eğitim verisi üzerinde FIT et ve TRANSFORM et
X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# Test verisini aynı dönüşümle TRANSFORM et
X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print("\n--- Model Eğitimi Başlıyor (Gerçek Veri Üzerinde) ---")
start_time = time.time()

model = RandomForestRegressor(
    n_estimators=150,      
    max_depth=12,          
    min_samples_leaf=8,    
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, Y_train)

end_time = time.time()
print(f"Eğitim Tamamlandı. Süre: {end_time - start_time:.2f} saniye.")

Y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)

print("\n--- Model Performans Değerlendirmesi (Test Seti) ---")
print(f"RMSE (Test Seti): {rmse:.4f}")
print(f"MAE (Test Seti): {mae:.4f}")

model_filename = 'retrained_occupancy_model.joblib'
scaler_filename = 'retrained_standard_scaler.joblib'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\nModel başarıyla yeniden eğitildi ve kaydedildi: {model_filename}")
print(f"Scaler başarıyla kaydedildi: {scaler_filename}")

Y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
train_mae = mean_absolute_error(Y_train, Y_train_pred)

print(f"RMSE (EĞİTİM Seti): {train_rmse:.4f}")
print(f"MAE (EĞİTİM Seti): {train_mae:.4f}")
print("overfit ihtimali" if train_mae< mae else "no problem")

plt.figure(figsize=(15, 6))

# Verileri çizmek için basit bir indeks (zaman serisi sırası) oluştur
test_indices = range(len(Y_test))

# Gerçek Değerler (Mavi)
plt.plot(test_indices, Y_test, label='Gerçek Doluluk Oranı (Test)', marker='.', linestyle='None', alpha=0.3)

# Tahmin Edilen Değerler (Kırmızı)
plt.plot(test_indices, Y_pred, label='Tahmin Edilen Doluluk Oranı (Test)', marker='.', linestyle='None', alpha=0.7)

plt.title(f'Gerçek vs. Tahmin Edilen Doluluk Oranı (Test Seti) - MAE: {mae:.4f}')
plt.xlabel('Test Veri Noktası Sırası (Zaman İçinde İlerler)')
plt.ylabel('Doluluk Oranı (Occupancy Ratio)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# 2. Hata (Residual) Dağılımı Grafiği
residuals = Y_test - Y_pred

plt.figure(figsize=(15, 6))

# Hataları dağılım grafiği olarak çizme
plt.scatter(Y_pred, residuals, alpha=0.5)

# Y=0 Çizgisi (Sıfır Hata Çizgisi)
plt.axhline(y=0, color='r', linestyle='--') 

plt.title(f'Hata (Residual) Dağılımı - Test Seti (MAE: {mae:.4f})')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hata (Gerçek - Tahmin)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()