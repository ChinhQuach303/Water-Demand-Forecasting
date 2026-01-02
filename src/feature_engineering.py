import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from config import setup_logging, DATA_DIR, TEST_SIZE, VAL_SIZE, set_seed

logger = setup_logging(__name__)

def create_features(df):
    logger.info("🔨 Creating features...")
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Is_Summer_Peak'] = ((df['Month'] >= 6) & (df['Month'] <= 8)).astype(int)
    
    if 'Temperature' in df.columns:
        df['Temp_mean_3m'] = df.groupby('PWSID')['Temperature'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['CDD'] = np.maximum(df['Temperature'] - 18, 0)
    
    df['lag_1'] = df.groupby('PWSID')['Demand'].shift(1)
    df['lag_12'] = df.groupby('PWSID')['Demand'].shift(12)
    df['rolling_mean_12'] = df.groupby('PWSID')['Demand'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['diff_12'] = df.groupby('PWSID')['Demand'].diff(12)
    
    if 'Temperature' in df.columns:
        df['Temp_lag_1'] = df.groupby('PWSID')['Temperature'].shift(1)
        df['Summer_Heat_Interaction'] = df['Is_Summer_Peak'] * df['CDD']
    if 'Precipitation' in df.columns:
        df['Precip_lag_1'] = df.groupby('PWSID')['Precipitation'].shift(1)
    return df.fillna(method='bfill').fillna(0)

def main():
    set_seed()
    logger.info("Starting feature engineering step...")
    
    input_path = os.path.join(DATA_DIR, 'preprocessed.parquet')
    if not os.path.exists(input_path):
        logger.error(f"❌ Input file not found: {input_path}. Run preprocess.py first.")
        return

    df = pd.read_parquet(input_path)
    df_features = create_features(df)
    
    # Split logic
    unique_dates = df_features['Date'].sort_values().unique()
    n_test, n_val = int(len(unique_dates)*TEST_SIZE), int(len(unique_dates)*VAL_SIZE)
    test_start, val_start = unique_dates[-n_test], unique_dates[-(n_test+n_val)]
    
    logger.info(f"Splitting data: Val Start: {val_start}, Test Start: {test_start}")

    train_df = df_features[df_features['Date'] < val_start].copy()
    val_df = df_features[(df_features['Date'] >= val_start) & (df_features['Date'] < test_start)].copy()
    test_df = df_features[df_features['Date'] >= test_start].copy()
    
    # Label Encoding PWSID
    le = LabelEncoder()
    train_df['PWSID_enc'] = le.fit_transform(train_df['PWSID'])
    # Handle new categories safely if possible, but for this pipeline we stick to original logic
    # Note: If val/test has unseen PWSID, this might crash. 
    # Original notebook assumed full dataset for PWSID check or consistent IDs. 
    # For now, we reuse fit logic if IDs are consistent across time (usually true for water sensors).
    val_df = val_df[val_df['PWSID'].isin(le.classes_)].copy()
    test_df = test_df[test_df['PWSID'].isin(le.classes_)].copy()
    
    val_df['PWSID_enc'] = le.transform(val_df['PWSID'])
    test_df['PWSID_enc'] = le.transform(test_df['PWSID'])
    
    train_path = os.path.join(DATA_DIR, 'train.parquet')
    val_path = os.path.join(DATA_DIR, 'val.parquet')
    test_path = os.path.join(DATA_DIR, 'test.parquet')
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    
    # Save encoder classes if needed generally, but for simplicity relying on retraining for now or simple pickle
    np.save(os.path.join(DATA_DIR, 'pwsid_classes.npy'), le.classes_)
    
    logger.info(f"✅ Feature engineering complete. Files saved: {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    main()
