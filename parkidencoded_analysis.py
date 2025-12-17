import pandas as pd
import os

# --- AYARLAR ---
# Veri setinizin dosya yolunu ve adını buraya girin!
FILE_PATH = '2020_Park_Doluluk_Tahmin_Tamamlandi (1).csv'

# Kolon Adları
ENCODED_ID_COL = 'park_id_encoded'
CAPACITY_COL = 'max_capacity'
# ----------------

def get_id_capacity_mapping(file_path: str, encoded_col: str, capacity_col: str):
    """
    Veri setini okur ve benzersiz kodlanmış ID'ler ile karşılık gelen
    maksimum kapasite değerlerini gösteren bir tablo oluşturur.
    """
    if not os.path.exists(file_path):
        print(f"❌ HATA: Dosya bulunamadı: {file_path}")
        return

    try:
        # Dosya formatını otomatik olarak algılayarak okuma
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("❌ HATA: Desteklenmeyen dosya formatı. Lütfen CSV veya Excel kullanın.")
            return

    except Exception as e:
        print(f"❌ HATA: Dosya okuma sırasında hata oluştu: {e}")
        return

    # Gerekli kolonların kontrolü
    if encoded_col not in df.columns or capacity_col not in df.columns:
        print(f"❌ HATA: Veri setinde '{encoded_col}' veya '{capacity_col}' kolonu bulunamadı.")
        print(f"Mevcut kolonlar: {df.columns.tolist()}")
        return

    # 1. Benzersiz Eşleşmeleri Çıkarma
    # Sadece bu iki kolonu seç ve tekrar eden satırları (aynı ID ve aynı kapasite) kaldır.
    mapping_df = df[[encoded_col, capacity_col]].drop_duplicates()
    
    # Kodlanmış ID'ye göre sıralama
    mapping_df = mapping_df.sort_values(by=encoded_col).reset_index(drop=True)
    
    # Kontrol: Bir ID'ye birden fazla kapasite atanmış mı?
    if mapping_df[encoded_col].duplicated().any():
        print("⚠️ KRİTİK UYARI: Bir 'park_id_encoded' değerine birden fazla 'max_capacity' değeri atanmış.")
        print("Modelinize tutarsız veri girilmiş olabilir. Veri setinizi kontrol edin.")
        # Sadece tutarsızlıkları göster
        inconsistent_ids = mapping_df[mapping_df[encoded_col].duplicated(keep=False)]
        print("\n--- TUTARSIZ KAYITLAR ---")
        print(inconsistent_ids.to_string(index=False))
        print("---------------------------\n")

    # --- SONUÇLARI YAZDIRMA ---
    print("\n" + "="*50)
    print("🅿️ OTOPARK KODU VE KAPASİTE EŞLEŞMESİ 🅿️")
    print("="*50)
    print(f"Toplam Benzersiz Otopark Sayısı: {len(mapping_df)}")
    
    print("\n--- Kodlanmış ID -> Maksimum Kapasite Tablosu (Firebase'e Eklenecek Statik Veri) ---")
    print(mapping_df.to_string(index=False, header=True))
    print("="*50 + "\n")


# Fonksiyonu çalıştırın
get_id_capacity_mapping(FILE_PATH, ENCODED_ID_COL, CAPACITY_COL)