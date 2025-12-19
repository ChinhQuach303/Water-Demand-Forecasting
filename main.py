"""
HYBRID ANCHOR-RESIDUAL FRAMEWORK - BACKTESTING EDITION (FIXED)
Architecture: Ridge Anchor + Tree Residual + Vectorized Fail-Safe Safety
"""

import numpy as np
import pandas as pd
import optuna # cần optuna để tối ưu tham số
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm # Progress bar
from sklearn.preprocessing import LabelEncoder, StandardScaler # stabdardscaler là z-transform, labelencoder để encode pwsid
from sklearn.linear_model import Ridge # dùng ridge, để regularzation tốt hơn linear, cần tối ưu tham số alpha
from sklearn.metrics import mean_absolute_error # mae
from optuna.samplers import TPESampler # ?

# Configuration
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============= CONFIGURATION =============
FILE_PATH = 'CaRDS.csv' # file đầu vào
N_FOLDS = 5 # số fold, cross-valdilation
RANDOM_SEED = 42
OPTUNA_TRIALS_PER_FOLD = 20 # tổng 100 trials 

# ============= SEEDING =============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============= HYBRID MODEL (CORE) =============
# hàm hybrid residual gồm rigde và xgboost
class HybridAnchorResidualModel:
    def __init__(self, ridge_alpha=1.0, residual_model='xgboost', residual_params=None, seed=42):
        self.ridge = Ridge(alpha=ridge_alpha, random_state=seed) # ridge
        self.scaler = StandardScaler() # scaler
        self.residual_model_name = residual_model # residual model
        self.residual_params = residual_params or {} # residual params
        self.seed = seed
        self.residual_model = None

    def _create_residual_model(self):
        common_params = {'random_state': self.seed, 'n_jobs': -1}
        if self.residual_model_name == 'xgboost':
            import xgboost as xgb
            params = {
                'objective': 'reg:quantileerror', 'quantile_alpha': 0.5,
                'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6,
                'subsample': 0.8, 'tree_method': 'hist', 'early_stopping_rounds': 100
            }
            params.update(common_params)
            params.update(self.residual_params)
            return xgb.XGBRegressor(**params)
        return None 

    def fit(self, X, y, sample_weight=None, eval_set=None):
        # 1. Ridge Anchor
        X_scaled = self.scaler.fit_transform(X) #X lúc này đã transform nên chưa chắc cần
        self.ridge.fit(X_scaled, y, sample_weight=sample_weight) # model ridge để fit
        y_pred_ridge = self.ridge.predict(X_scaled) # predict cho x_scaled
        residuals = y - y_pred_ridge # residual, hàm lỗi

        # 2. Residual Model
        self.residual_model = self._create_residual_model()
        
        eval_set_res = []
        # áp dụng với tập val để học
        if eval_set:
            for X_v, y_v in eval_set:
                X_v_sc = self.scaler.transform(X_v) # lại transform ?
                r_v = y_v - self.ridge.predict(X_v_sc) # ok predict ridge cho tập val sau đó tạo 1 cái list để lưu X_val, y_val 
                eval_set_res.append((X_v, r_v))
        # resudual_moedl cho X(train)
        self.residual_model.fit(X, residuals, sample_weight=sample_weight, 
                                eval_set=eval_set_res, verbose=False)
        return self
    # oreddict sẽ là predict(x_scaled)+ ressidual(X)
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.ridge.predict(X_scaled) + self.residual_model.predict(X)

# ============= SAFETY LAYER (FAIL-SAFE) =============


# cơ chế bao gồm 2 hàm floor và buffer, tạo sàn và trần sau đó cộng dần cho từng điểm dựa theo các tham số
class WaterDemandSafetyLayer:
    def __init__(self, config):
        self.config = config
        self.risk_profile = None
        self.global_median_lag1 = 0 # biến này dùng khi mà mất lag_1
    # 
    def fit(self, df_calib, y_true, y_pred_raw):
        self.global_median_lag1 = df_calib['lag_1'].median()
        
        analysis = df_calib[['PWSID_enc', 'Month']].copy()
        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        analysis['Shortage'] = y_true_arr - y_pred_raw
        
        grp_std = analysis.groupby('PWSID_enc')['Shortage'].std().rename('Error_Std')
        
        shortage_only = analysis[analysis['Shortage'] > 0]
        grp_max = shortage_only.groupby('PWSID_enc')['Shortage'].max().rename('Max_Shortage')
        
        summer_m = self.config['summer_months']
        summer_s = shortage_only[shortage_only['Month'].isin(summer_m)]
        grp_max_summer = summer_s.groupby('PWSID_enc')['Shortage'].max().rename('Max_Summer_Shortage')
        
        self.risk_profile = pd.concat([grp_std, grp_max, grp_max_summer], axis=1).fillna(0)
        return self
    
    def predict(self, raw_pred, df_context):
        # 1. Fail-Safe Detection
        current_lag1 = df_context['lag_1'].values
        mask_sensor_fail = np.isclose(current_lag1, self.global_median_lag1, atol=1e-3)
        
        # 2. Floor Logic
        cfg_f = self.config['floor']
        lag_12 = df_context['lag_12'].values
        lag_1 = df_context['lag_1'].values
        
        floor_yoy = lag_12 * cfg_f['yoy_growth_min']
        
        mom_factor = np.full_like(raw_pred, cfg_f['mom_drop_max'])
        mom_factor[df_context['Month'].isin(self.config['summer_months'])] = cfg_f['mom_drop_summer']
        
        floor_mom = lag_1 * mom_factor
        floor_mom = np.nan_to_num(floor_mom, nan=0.0)
        
        floored_pred = np.maximum(raw_pred, floor_yoy)
        floored_pred = np.maximum(floored_pred, floor_mom)
        
        # 3. Buffer Logic
        cfg_b = self.config['buffer']
        risk = self.risk_profile.reindex(df_context['PWSID_enc']).fillna(0)
        
        buf_base = risk['Error_Std'].values * cfg_b['base_sigma']
        buf_hist = risk['Max_Shortage'].values * cfg_b['hist_coverage']
        
        raw_buffer = np.maximum(buf_base, buf_hist)
        
        # === FAIL SAFE INJECTION ===
        fail_safe_add = np.zeros_like(raw_buffer)
        fail_safe_add[mask_sensor_fail] = risk['Error_Std'].values[mask_sensor_fail] * 1.5
        
        total_buffer = raw_buffer + fail_safe_add
        
        # Cap logic
        cap_val = floored_pred * cfg_b['max_cap_pct']
        cap_val[mask_sensor_fail] *= 1.5 
        
        final_buffer = np.minimum(total_buffer, cap_val)
        return floored_pred + final_buffer

# ============= DATA UTILS (FIXED) =============
# clean base, thay thê các giá trị chặn trên và dưới  = NaN sau đó interpolate bằng linear
def clean_physics_based(series):
    """Helper to clean demand data"""
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
# load file, melt table rồi đổi tên sau đó tổng hợp các feature theo tháng, deman tổng theo tháng, precip, temperature thì lấy trung bình
def get_data(file_path):
    """Load and PROCESS raw data (Melt + Pivot) to create Date column."""
    if not os.path.exists(file_path):
        print("⚠️ Demo Mode: Creating synthetic data...")
        dates = pd.date_range(start='2010-01-01', end='2020-12-31', freq='M')
        pwsids = ['Station_A', 'Station_B', 'Station_C']
        data = []
        for p in pwsids:
            base = 1000 + np.random.rand() * 500
            trend = np.linspace(0, 200, len(dates))
            season = np.sin(np.arange(len(dates)) / 6) * 300
            vals = base + trend + season
            vals[::20] = np.median(vals) # Failures
            for d, v in zip(dates, vals):
                data.append([p, 'Demand', d, v])
                data.append([p, 'Temperature', d, 25])
        df = pd.DataFrame(data, columns=['PWSID', 'Variable', 'Date', 'Value'])
        df = df.pivot_table(index=['PWSID', 'Date'], columns='Variable', values='Value').reset_index()
        return df
    
    else:
        # ✅ RESTORED FULL PROCESSING LOGIC HERE
        print(f"📂 Loading and Processing {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # 1. Standardize Variable names
            df['Variable'] = df['Variable'].astype(str).str.strip().str.lower()
            
            # 2. Identify Date Columns (Wide format to Long)
            date_cols = [c for c in df.columns if c not in ['PWSID', 'Variable']]
            
            # 3. Melt
            df_melt = df.melt(id_vars=['PWSID', 'Variable'], value_vars=date_cols, 
                              var_name='Date', value_name='Value')
            
            # 4. Pivot
            df_pivot = df_melt.pivot_table(index=['PWSID', 'Date'], columns='Variable', values='Value').reset_index()
            
            # 5. Rename & Convert types
            rename_map = {'demand': 'Demand', 'temperature': 'Temperature', 
                          'precipitation': 'Precipitation', 'pdsi': 'PDSI'}
            df_pivot.rename(columns=rename_map, inplace=True)
            df_pivot['Date'] = pd.to_datetime(df_pivot['Date'])
            
            # 6. Fill missing values
            for col in ['Temperature', 'Precipitation', 'PDSI']:
                if col in df_pivot.columns:
                    val = 0 if col == 'Precipitation' else df_pivot[col].median()
                    df_pivot[col] = df_pivot[col].fillna(val)
            
            # 7. Clean Demand
            if 'Demand' in df_pivot.columns:
                df_pivot['Demand'] = df_pivot.groupby('PWSID')['Demand'].transform(clean_physics_based)
                df_pivot['Demand'] = df_pivot['Demand'].fillna(0)
            
            # 8. Final Sort
            df_final = df_pivot.sort_values(['PWSID', 'Date']).reset_index(drop=True)
            return df_final
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            raise e
# tạo các feature, 9 feature cần tạo bao gồm, year, month, pwsid_enc, cdd, lag_1, lag_12, summer_heat_interaction, precip_lag, diff_12
def feature_eng(df):
    """Create lag features"""
    # Now df is guaranteed to have 'Date' and 'PWSID'
    df = df.sort_values(['PWSID', 'Date']).copy()
    df['Month'] = df['Date'].dt.month
    df['lag_1'] = df.groupby('PWSID')['Demand'].shift(1)
    df['lag_12'] = df.groupby('PWSID')['Demand'].shift(12)
    df['diff_12'] = df.groupby('PWSID')['Demand'].diff(12)
    return df.bfill().fillna(0) 

# ============= OPTIMIZATION INSIDE BACKTEST =============
def tune_safety_params(df_calib, y_true, y_pred_raw, seed):
    """Mini Optuna run for each fold"""
    
    def objective(trial):
        cfg = {
            'summer_months': [6, 7, 8],
            'floor': {
                'yoy_growth_min': trial.suggest_float('yoy', 1.0, 1.15),
                'mom_drop_max': trial.suggest_float('mom', 0.85, 0.95),
                'mom_drop_summer': 0.98,
                'mom_drop_fall': 0.9
            },
            'buffer': {
                'base_sigma': trial.suggest_float('sigma', 1.0, 2.5),
                'hist_coverage': trial.suggest_float('hist', 0.8, 1.2),
                'max_cap_pct': trial.suggest_float('cap', 0.2, 0.5)
            }
        }
        layer = WaterDemandSafetyLayer(cfg)
        layer.fit(df_calib, y_true, y_pred_raw)
        preds = layer.predict(y_pred_raw, df_calib)
        
        diff = preds - y_true
        shortage = np.mean(np.maximum(-diff, 0))
        surplus = np.mean(np.maximum(diff, 0))
        
        return surplus + (shortage * 50)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_FOLD, show_progress_bar=False)
    
    best = study.best_params
    return {
        'summer_months': [6, 7, 8],
        'floor': {'yoy_growth_min': best['yoy'], 'mom_drop_max': best['mom'], 'mom_drop_summer': 0.98, 'mom_drop_fall': 0.9},
        'buffer': {'base_sigma': best['sigma'], 'hist_coverage': best['hist'], 'max_cap_pct': best['cap']}
    }

# ============= BACKTEST ENGINE =============
def run_backtest():
    set_seed(RANDOM_SEED)
    
    # 1. Load & Prep
    try:
        df_raw = get_data(FILE_PATH)
        if df_raw is None: return
        df = feature_eng(df_raw)
    except Exception as e:
        print(f"❌ Critical Data Error: {e}")
        return
    
    # Encode PWSID
    le = LabelEncoder()
    df['PWSID_enc'] = le.fit_transform(df['PWSID'])
    
    # 2. Time Splitting
    unique_dates = df['Date'].sort_values().unique()
    
    if len(unique_dates) < N_FOLDS + 2:
        print("❌ Not enough dates for 5-fold backtest.")
        return

    # Split dates into chunks
    chunks = np.array_split(unique_dates, N_FOLDS + 1)
    
    metrics = []
    all_predictions = []
    
    print(f"🚀 Starting Backtest with {N_FOLDS} Folds on {len(df)} rows...")
    
    current_train_dates = chunks[0]
    
    for i in tqdm(range(1, N_FOLDS + 1), desc="Processing Folds"):
        test_dates = chunks[i]
        
        # --- A. Data Slicing ---
        train_full_df = df[df['Date'].isin(current_train_dates)].copy()
        test_df = df[df['Date'].isin(test_dates)].copy()
        
        if len(train_full_df) == 0 or len(test_df) == 0:
            continue

        # Internal Split for Calibration (Safety Layer needs unseen data)
        calib_cutoff_idx = int(len(current_train_dates) * 0.8)
        if calib_cutoff_idx == 0: calib_cutoff_idx = 1 # Prevent empty train
        
        train_dates_inner = current_train_dates[:calib_cutoff_idx]
        calib_dates_inner = current_train_dates[calib_cutoff_idx:]
        
        inner_train_df = df[df['Date'].isin(train_dates_inner)]
        inner_calib_df = df[df['Date'].isin(calib_dates_inner)]
        
        features = ['PWSID_enc', 'Month', 'lag_1', 'lag_12', 'diff_12']
        
        # --- B. Train Hybrid Model ---
        model = HybridAnchorResidualModel(seed=RANDOM_SEED)
        model.fit(
            inner_train_df[features], inner_train_df['Demand'],
            eval_set=[(inner_calib_df[features], inner_calib_df['Demand'])]
        )
        
        # --- C. Optimize Safety Layer ---
        pred_calib_raw = model.predict(inner_calib_df[features])
        
        best_safety_cfg = tune_safety_params(
            inner_calib_df, inner_calib_df['Demand'], pred_calib_raw, seed=RANDOM_SEED + i
        )
        
        safety_layer = WaterDemandSafetyLayer(best_safety_cfg)
        safety_layer.fit(inner_calib_df, inner_calib_df['Demand'], pred_calib_raw)
        
        # --- D. Final Prediction on Test ---
        pred_test_raw = model.predict(test_df[features])
        pred_test_safe = safety_layer.predict(pred_test_raw, test_df)
        
        # --- E. Store Results ---
        res_df = test_df[['Date', 'PWSID', 'Demand']].copy()
        res_df['Predicted'] = pred_test_safe
        res_df['Raw_Pred'] = pred_test_raw
        res_df['Fold'] = i
        all_predictions.append(res_df)
        
        mae = mean_absolute_error(test_df['Demand'], pred_test_safe)
        shortage = np.mean(np.maximum(test_df['Demand'] - pred_test_safe, 0))
        metrics.append({'Fold': i, 'MAE': mae, 'Shortage': shortage})
        
        # Update Window
        current_train_dates = np.concatenate([current_train_dates, test_dates])

    # ============= VISUALIZATION =============
    if not all_predictions:
        print("❌ No predictions generated.")
        return

    results_df = pd.concat(all_predictions)
    metrics_df = pd.DataFrame(metrics)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(metrics_df)
    print(f"\nAverage MAE: {metrics_df['MAE'].mean():.2f}")
    
    # Plot 1: Continuous Prediction
    agg_res = results_df.groupby('Date')[['Demand', 'Predicted']].sum().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(agg_res['Date'], agg_res['Demand'], 'k-', label='Actual Demand', alpha=0.6)
    plt.plot(agg_res['Date'], agg_res['Predicted'], 'r--', label='Backtest Prediction (Safe)', alpha=0.8)
    
    plt.title('Backtest: Walk-Forward Validation Results (Concatenated Folds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot 2: Metrics Stability
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE', color='tab:blue')
    ax1.plot(metrics_df['Fold'], metrics_df['MAE'], 'o-', color='tab:blue', label='MAE')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Shortage', color='tab:red')
    ax2.bar(metrics_df['Fold'], metrics_df['Shortage'], color='tab:red', alpha=0.3, label='Shortage')
    
    plt.title('Model Stability Across Folds (Error & Risk)')
    plt.show()

if __name__ == "__main__":
    run_backtest()