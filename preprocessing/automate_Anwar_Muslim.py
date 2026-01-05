import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- KONFIGURASI PATH ---
# Pastikan path ini menunjuk ke file CSV asli Anda
INPUT_PATH = "C:/Users/User/Documents/Eksperimen_SML_Anwar-Muslim/taxi_tripdata_raw/taxi_tripdata_raw.csv"
OUTPUT_FOLDER = "taxi_tripdata_preprocessing"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")
    print(f"[INFO] Memuat data dari {path}...")
    return pd.read_csv(path)

def preprocess_data(df):
    print("[INFO] Melakukan Preprocessing...")
    initial_shape = df.shape
    
    # 1. Konversi Datetime (Wajib untuk hitung durasi)
    # Cek nama kolom di data asli: 'lpep_pickup_datetime' atau 'tpep_pickup_datetime'
    if 'lpep_pickup_datetime' in df.columns:
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    elif 'tpep_pickup_datetime' in df.columns:
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:
        raise ValueError("Kolom datetime tidak ditemukan! Cek nama header di CSV.")

    df[pickup_col] = pd.to_datetime(df[pickup_col])
    df[dropoff_col] = pd.to_datetime(df[dropoff_col])
    
    # 2. Hitung Durasi (Menit) - Ini kunci agar R2 Score tinggi
    df['trip_duration'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    
    # 3. Definisi Fitur & Target
    target_col = 'fare_amount'
    feature_cols = ['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID', 'trip_duration']
    
    # 4. Ambil Kolom Relevan
    relevant_cols = feature_cols + [target_col]
    
    # Pastikan kolom ada semua
    missing_cols = [c for c in relevant_cols if c not in df.columns]
    if missing_cols:
        print(f"[WARNING] Kolom berikut hilang dari data: {missing_cols}")
        # Lanjut hanya dengan kolom yang ada (safety measure)
        relevant_cols = [c for c in relevant_cols if c in df.columns]

    df = df[relevant_cols].copy()
    
    # 5. Cleaning Rows (Sekarang aman karena kolom sampah sudah dibuang)
    df = df.dropna()
    
    # Filter Logis (Hapus data ngawur)
    if 'fare_amount' in df.columns and 'trip_distance' in df.columns and 'trip_duration' in df.columns:
        df = df[
            (df['fare_amount'] > 0) & 
            (df['trip_distance'] > 0) & 
            (df['trip_duration'] > 0) & 
            (df['trip_duration'] <= 180) # Max 3 jam
        ]
        
    # Filter Passenger (Optional)
    if 'passenger_count' in df.columns:
        df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
        df['passenger_count'] = df['passenger_count'].astype(int)
        
    print(f"[INFO] Cleaning selesai. {initial_shape} -> {df.shape}")
    
    if df.empty:
        raise ValueError("Data Kosong setelah cleaning! Cek logika filter Anda.")
        
    return df

def save_split_data(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Cek target
    target = 'fare_amount'
    if target not in df.columns:
        raise ValueError("Target 'fare_amount' hilang saat preprocessing.")
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Gabung kembali untuk disimpan (Format MLflow)
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    train_path = os.path.join(output_folder, 'train.csv')
    test_path = os.path.join(output_folder, 'test.csv')
    
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
    
    print(f"[SUKSES] Data tersimpan di folder '{output_folder}'")
    print(f" - Train shape: {train_set.shape}")
    print(f" - Test shape : {test_set.shape}")

if __name__ == "__main__":
    try:
        df = load_data(INPUT_PATH)
        df_clean = preprocess_data(df)
        save_split_data(df_clean, OUTPUT_FOLDER)
    except Exception as e:
        print(f"[ERROR] {e}")