import pandas as pd
import numpy as np
import os
from config import setup_logging, FILE_PATH, DATA_DIR, set_seed

logger = setup_logging(__name__)

def clean_physics_based(series):
    """Giữ nguyên logic làm sạch dữ liệu ban đầu"""
    median_val = series.median()
    if pd.isna(median_val) or median_val <= 0:
        return series.fillna(0)
    phys_min = median_val * 0.05
    phys_max = median_val * 10.0
    mask_invalid = (series < phys_min) | (series > phys_max)
    if mask_invalid.any():
        series_clean = series.copy()
        series_clean[mask_invalid] = np.nan
        return series_clean.interpolate(method='linear', limit_direction='both')
    return series

def load_and_process_data(file_path):
    logger.info(f"📂 Loading data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"❌ Error: File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"❌ Error reading file: {e}")
        return None
    
    df['Variable'] = df['Variable'].astype(str).str.strip().str.lower()
    date_cols = [c for c in df.columns if c not in ['PWSID', 'Variable']]
    df_melt = df.melt(id_vars=['PWSID', 'Variable'], value_vars=date_cols, 
                      var_name='Date', value_name='Value')
    df_pivot = df_melt.pivot_table(index=['PWSID', 'Date'], columns='Variable', values='Value').reset_index()
    
    rename_map = {'demand': 'Demand', 'temperature': 'Temperature', 'precipitation': 'Precipitation', 'pdsi': 'PDSI'}
    df_pivot.rename(columns=rename_map, inplace=True)
    df_pivot['Date'] = pd.to_datetime(df_pivot['Date'])
    df_final = df_pivot.sort_values(['PWSID', 'Date']).reset_index(drop=True)
    
    for col in ['Temperature', 'Precipitation', 'PDSI']:
        if col in df_final.columns:
            val = 0 if col == 'Precipitation' else df_final[col].median()
            df_final[col] = df_final[col].fillna(val)
    if 'Demand' in df_final.columns:
        df_final['Demand'] = df_final.groupby('PWSID')['Demand'].transform(clean_physics_based)
        df_final['Demand'] = df_final['Demand'].fillna(0)
    
    return df_final

if __name__ == "__main__":
    set_seed()
    logger.info("Starting preprocessing step...")
    df = load_and_process_data(FILE_PATH)
    if df is not None:
        output_path = os.path.join(DATA_DIR, 'preprocessed.parquet')
        df.to_parquet(output_path)
        logger.info(f"✅ Preprocessing complete. Data saved to {output_path}")
    else:
        logger.error("❌ Preprocessing failed.")
