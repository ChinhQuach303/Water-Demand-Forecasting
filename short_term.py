"""
HYBRID ANCHOR-RESIDUAL FRAMEWORK WITH ENHANCED EVALUATION
Architecture: Ridge Anchor + Tree Residual + Vectorized Risk-Based Safety
Status: UPDATED WITH DETAILED EVALUATION METRICS
"""

import numpy as np
import pandas as pd
import optuna
import os
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from optuna.samplers import TPESampler 

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid")

# ============= CONFIGURATION =============
FILE_PATH = 'CaRDS.csv'
TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_SEED = 42

# ============= SEEDING FUNCTION =============
def set_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 Global Seed set to: {seed}")

# ============= HYBRID ANCHOR-RESIDUAL MODEL =============
class HybridAnchorResidualModel:
    """Ridge Anchor + Tree-based Residual Learning"""
    
    def __init__(self, ridge_alpha=1.0, residual_model='xgboost', residual_params=None, seed=42):
        self.ridge_alpha = ridge_alpha
        self.residual_model_name = residual_model
        self.residual_params = residual_params or {}
        self.seed = seed
        
        self.ridge = Ridge(alpha=ridge_alpha, random_state=seed)
        self.scaler = StandardScaler()
        self.residual_model = None
        
    def _create_residual_model(self):
        """Create residual model (XGBoost Only)"""
        import xgboost as xgb
        default_params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'n_estimators': 3000,
            'learning_rate': 0.02,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.seed,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 200
        }
        default_params.update(self.residual_params)
        return xgb.XGBRegressor(**default_params)
    
    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=False):
        """Two-stage training"""
        print(f"   ⚓ Fitting Anchor Ridge Layer...")
        
        # Stage 1: Ridge
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y, sample_weight=sample_weight)
        
        y_pred_ridge = self.ridge.predict(X_scaled)
        residuals = y - y_pred_ridge
        
        ridge_mae = mean_absolute_error(y, y_pred_ridge)
        print(f"      Ridge MAE: {ridge_mae:,.0f}")
        
        # Stage 2: Tree on residuals
        print(f"   🚀 Fitting {self.residual_model_name.upper()} on Residuals...")
        self.residual_model = self._create_residual_model()
        
        # Prepare eval_set for residuals
        eval_set_residual = None
        if eval_set:
            eval_set_residual = []
            for X_val, y_val in eval_set:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_ridge = self.ridge.predict(X_val_scaled)
                residual_val = y_val - y_val_ridge
                eval_set_residual.append((X_val, residual_val))
        
        # Fit (XGBoost)
        self.residual_model.fit(
            X, residuals,
            sample_weight=sample_weight,
            eval_set=eval_set_residual,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X):
        """Combine predictions"""
        X_scaled = self.scaler.transform(X)
        pred_ridge = self.ridge.predict(X_scaled)
        pred_residual = self.residual_model.predict(X)
        return pred_ridge + pred_residual

# ============= SAFETY LAYER V4.0 =============
class WaterDemandSafetyLayer:
    """Vectorized Risk-Based Safety Layer với cơ chế FAIL-SAFE."""
    
    def __init__(self, config):
        self.config = config
        self.risk_profile = None
        self.global_median_lag1 = 0
    
    def fit(self, df_val, y_true, y_pred_raw):
        """Learn risk profile (vectorized)"""
        self.global_median_lag1 = df_val['lag_1'].median()
        
        analysis = df_val[['PWSID_enc', 'Month']].copy()
        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        analysis['Shortage'] = y_true_arr - y_pred_raw
        
        # 1. Error Std per station
        grp_std = analysis.groupby('PWSID_enc')['Shortage'].std().rename('Error_Std')
        
        # 2. Max Historical Shortage
        shortage_only = analysis[analysis['Shortage'] > 0]
        grp_max = shortage_only.groupby('PWSID_enc')['Shortage'].max().rename('Max_Shortage')
        
        # 3. Max Summer Shortage
        summer_months = self.config['summer_months']
        summer_shortage = shortage_only[shortage_only['Month'].isin(summer_months)]
        grp_max_summer = summer_shortage.groupby('PWSID_enc')['Shortage'].max().rename('Max_Summer_Shortage')
        
        self.risk_profile = pd.concat([grp_std, grp_max, grp_max_summer], axis=1).fillna(0)
        self.risk_profile.index.name = 'PWSID_enc'
        
        return self
    
    def predict(self, raw_pred, df_context, explain=False):
        """Apply vectorized safety adjustments with FAIL-SAFE CHECK"""
        
        # --- 0. FAIL-SAFE DETECTION ---
        current_lag1 = df_context['lag_1'].values
        mask_sensor_failure = np.isclose(current_lag1, self.global_median_lag1, atol=1e-3)
        
        # --- A. Smart Floor ---
        cfg_f = self.config['floor']
        lag_12 = df_context['lag_12'].values
        lag_1 = df_context['lag_1'].values
        months = df_context['Month'].values
        
        floor_yoy = lag_12 * cfg_f['yoy_growth_min']
        
        mom_factor = np.full_like(raw_pred, cfg_f['mom_drop_max'])
        mask_summer = np.isin(months, self.config['summer_months'])
        mom_factor[mask_summer] = cfg_f['mom_drop_summer']
        mask_fall = np.isin(months, [9, 10, 11])
        mom_factor[mask_fall] = cfg_f['mom_drop_fall']
        
        floor_mom = lag_1 * mom_factor
        mask_nan = np.isnan(floor_mom)
        floor_mom[mask_nan] = floor_yoy[mask_nan]
        
        floored_pred = np.maximum(raw_pred, floor_yoy)
        floored_pred = np.maximum(floored_pred, floor_mom)
        
        # --- B. Adaptive Risk Buffer ---
        if not self.config['buffer']['enabled']:
            final_pred = floored_pred
            buffer_vals = np.zeros_like(floored_pred)
        else:
            cfg_b = self.config['buffer']
            pwsids = df_context['PWSID_enc']
            risk_vec = self.risk_profile.reindex(pwsids).fillna(0)
            
            buf_base = risk_vec['Error_Std'].values * cfg_b['base_sigma']
            buf_hist = risk_vec['Max_Shortage'].values * cfg_b['hist_coverage']
            
            buf_summer = np.zeros_like(buf_base)
            buf_summer[mask_summer] = risk_vec['Max_Summer_Shortage'].values[mask_summer] * cfg_b['summer_coverage']
            
            raw_buffer = np.maximum(buf_base, buf_hist)
            raw_buffer = np.maximum(raw_buffer, buf_summer)
            
            # === C. FAIL-SAFE INJECTION ===
            fail_safe_add = np.zeros_like(raw_buffer)
            fail_safe_add[mask_sensor_failure] = risk_vec['Error_Std'].values[mask_sensor_failure] * 1.5 
            
            total_buffer = raw_buffer + fail_safe_add
            
            # Cap buffer
            cap_val = floored_pred * cfg_b['max_cap_pct']
            cap_val[mask_sensor_failure] = cap_val[mask_sensor_failure] * 1.5
            
            final_buffer = np.minimum(total_buffer, cap_val)
            
            final_pred = floored_pred + final_buffer
            buffer_vals = final_buffer
        
        if explain:
            expl_df = pd.DataFrame({
                'Raw': raw_pred,
                'Floored': floored_pred,
                'Buffer': buffer_vals,
                'Sensor_Fail': mask_sensor_failure, 
                'Final': final_pred
            }, index=df_context.index)
            return final_pred, expl_df
        
        return final_pred

# ============= DRIFT ADAPTER & MONITOR =============
class RobustLabelEncoder:
    """Safely handles unseen labels during inference (Production Ready)"""
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.unknown_token = '<UNK>'
    
    def fit(self, y):
        # Add string type conversion for safety
        y_str = np.array(y).astype(str)
        # Append unknown token to fit clean classes + 1 placeholder
        self.encoder.fit(np.append(y_str, [self.unknown_token]))
        self.classes_ = self.encoder.classes_
        return self
    
    def transform(self, y):
        y_str = np.array(y).astype(str)
        # Identify unseen
        mask_unknown = ~np.isin(y_str, self.classes_)
        
        # Replace unseen with unknown token
        y_safe = y_str.copy()
        y_safe[mask_unknown] = self.unknown_token
        
        return self.encoder.transform(y_safe)
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class DriftAdapter:
    """Manages Adaptive Learning: Sample Weights, Dynamic Windows & Monitoring"""
    
    def __init__(self, decay_rate=0.995):
        self.decay_rate = decay_rate
        
    def calculate_sample_weights(self, df, date_col='Date'):
        """
        Exponential Decay: Weight = decay ^ (months_ago)
        Recent data gets weight ~1.0, data 5 years ago gets much less.
        """
        dates = pd.to_datetime(df[date_col])
        max_date = dates.max()
        
        # Calculate months difference
        # (Approximate by days / 30)
        diff_months = (max_date - dates).dt.days / 30.0
        
        weights = np.power(self.decay_rate, diff_months)
        
        # Normalize weights so mean is 1.0 (to keep loss scale similar)
        weights = weights / weights.mean()
        
        print(f"   ⚖️ Sample Weights Applied: Decay={self.decay_rate}")
        print(f"      Min Weight: {weights.min():.4f} (Oldest)")
        print(f"      Max Weight: {weights.max():.4f} (Newest)")
        
        return weights.values

    def get_dynamic_risk_window(self, val_df, y_val, raw_pred_val, window_months=12):
        """
        Selects only the most recent 'window_months' for Safety Calibration.
        This adapts the risk profile to recent errors (Drift).
        """
        dates = pd.to_datetime(val_df['Date'])
        max_date = dates.max()
        cutoff_date = max_date - pd.DateOffset(months=window_months)
        
        mask = dates >= cutoff_date
        
        print(f"   🔄 Dynamic Safety Window: Last {window_months} months")
        print(f"      Selected {mask.sum()} samples from {len(val_df)} total validation samples.")
        
        return val_df[mask].copy(), y_val[mask], raw_pred_val[mask]

    def monitor_drift(self, y_true_recent, y_pred_recent, mae_threshold=None, shortage_threshold=None):
        """
        Checks recent performance using MAE and Shortage Volume.
        """
        if len(y_true_recent) == 0:
            return False, {}

        y_t = np.array(y_true_recent)
        y_p = np.array(y_pred_recent)
        
        # 1. MAE
        mae = mean_absolute_error(y_t, y_p)
        
        # 2. Shortage (Avg Volume of Shortage)
        diff = y_t - y_p
        shortage_vol = np.mean(np.maximum(diff, 0))
        
        print(f"   🔎 Drift Monitor -> MAE: {mae:.2f} (Lim: {mae_threshold:.2f}) | Shortage: {shortage_vol:.2f} (Lim: {shortage_threshold:.2f})")
        
        drift = False
        if mae_threshold is not None and mae > mae_threshold:
            print(f"   ⚠️  Drift Detected: High MAE.")
            drift = True
            
        if shortage_threshold is not None and shortage_vol > shortage_threshold:
            print(f"   ⚠️  Drift Detected: High Shortage.")
            drift = True
            
        if not drift:
             print("   ✅ Error within limits. Stability Maintained.")
            
        return drift, {'mae': mae, 'shortage': shortage_vol}


# ============= DATA PROCESSING =============
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
    print(f"\n📂 Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found.")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
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

def create_features(df):
    print("\n🔨 Creating features...")
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

# ============= SAFETY OPTIMIZATION =============
# ============= SAFETY OPTIMIZATION =============
def optimize_safety_optuna(val_df, y_val, raw_pred_val, n_trials=100, seed=42):
    """Optuna optimization with FIXED SEED"""
    print("\n" + "="*70)
    print(f"🚀 STARTING OPTUNA OPTIMIZATION ({n_trials} trials) - BALANCED MODE | SEED={seed}")
    print("="*70)
    
    mean_demand = np.mean(y_val)
    # Relaxed Shortage Constraint
    DYNAMIC_S_MAX = mean_demand * 0.02 # Allow slightly more shortage volume per instance
    target_under_rate = 0.03 # Tighten back to 3% to correct high shortage (was 5%)
    
    # print(f" ℹ️ Auto-tuning S_MAX constraint to: {DYNAMIC_S_MAX:,.0f}")
    
    val_df_sim = val_df.copy()
    val_df_sim['lag_1'] = np.nan
    
    def objective(trial):
        cfg = {
            'summer_months': [6, 7, 8],
            'floor': {
                'enabled': True,
                # Allow NEGATIVE growth (0.8 = 20% drop) to reduce surplus
                'yoy_growth_min': trial.suggest_float('yoy_min', 0.8, 1.1),
                'mom_drop_max': trial.suggest_float('mom_max', 0.80, 0.99),
                'mom_drop_summer': trial.suggest_float('mom_summer', 0.90, 1.05),
                'mom_drop_fall': trial.suggest_float('mom_fall', 0.70, 0.95)
            },
            'buffer': {
                'enabled': True,
                # Increase sigma to fix Shortagespike (User Request)
                'base_sigma': trial.suggest_float('base_sigma', 0.5, 2.5),
                'hist_coverage': trial.suggest_float('hist_cov', 0.5, 1.3),
                'summer_coverage': trial.suggest_float('summer_cov', 0.8, 1.6),
                'max_cap_pct': trial.suggest_float('max_cap', 0.05, 0.3)
            }
        }
        
        layer = WaterDemandSafetyLayer(cfg)
        layer.fit(val_df_sim, y_val, raw_pred_val)
        preds = layer.predict(raw_pred_val, val_df_sim)
        
        diff = preds - y_val
        
        # Metrics
        surplus_score = np.mean(np.maximum(diff, 0)) / mean_demand
        u_rate = np.mean(diff < 0)
        s_vol = np.mean(np.maximum(-diff, 0))
        
        # New: Add MAE to objective to encourage accuracy, not just safety
        mae_score = mean_absolute_error(y_val, preds) / mean_demand
        
        penalty = 0
        
        # Soft Penalty for Shortage Rate
        if u_rate > target_under_rate:
            # Linear penalty instead of massive exponential
            penalty += (u_rate - target_under_rate) * 50 
            
        # Soft Penalty for Shortage Volume
        if s_vol > DYNAMIC_S_MAX:
            penalty += (s_vol - DYNAMIC_S_MAX) / mean_demand * 10
        
        # Objective: Min(Surplus + MAE) subject to soft shortage constraints
        return surplus_score + mae_score + penalty
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Suppress Optuna logging to screen
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best = study.best_params
    final_config = {
        'summer_months': [6, 7, 8],
        'floor': {
            'enabled': True,
            'yoy_growth_min': best['yoy_min'],
            'mom_drop_max': best['mom_max'],
            'mom_drop_summer': best['mom_summer'],
            'mom_drop_fall': best['mom_fall']
        },
        'buffer': {
            'enabled': True,
            'base_sigma': best['base_sigma'],
            'hist_coverage': best['hist_cov'],
            'summer_coverage': best['summer_cov'],
            'max_cap_pct': best['max_cap']
        }
    }
    
    print(f"\n✅ Optimization complete! Best score: {study.best_value:.6f}")
    return final_config

# ============= ĐÁNH GIÁ CHI TIẾT =============
def detailed_evaluation(y_true, y_pred, name):
    """
    Tính toán metrics trung bình trên mỗi mẫu:
    MAE, R2, % Shortage, Avg Shortage Vol, Avg Surplus Vol
    """
    y_true_arr = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()
    diff = y_true_arr - y_pred_arr
    total_samples = len(y_true_arr)
    
    # 1. Accuracy
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    # 2. Shortage (Thực tế > Dự báo)
    shortage_mask = diff > 0
    shortage_count = np.sum(shortage_mask)
    shortage_pct = (shortage_count / total_samples) * 100
    # Lượng thiếu hụt trung bình trên TỔNG số mẫu
    avg_shortage_vol = np.sum(diff[shortage_mask]) / total_samples
    
    # 3. Surplus (Dự báo > Thực tế)
    surplus_mask = diff < 0
    # Lượng dư thừa trung bình trên TỔNG số mẫu
    avg_surplus_vol = np.sum(np.abs(diff[surplus_mask])) / total_samples
    
    metrics = {
        'Dataset': name,
        'MAE': mae,
        'R2': r2,
        'Shortage_%': shortage_pct,
        'Avg_Shortage': avg_shortage_vol,
        'Avg_Surplus': avg_surplus_vol
    }
    
    print(f"\n📊 RESULTS FOR: {name}")
    print(f"--------------------------------------")
    print(f"📈 MAE              : {mae:,.2f}")
    print(f"📈 R2 Score         : {r2:.4f}")
    print(f"🔴 Shortage %       : {shortage_pct:.2f}%")
    print(f"🔴 Avg Shortage Vol : {avg_shortage_vol:,.2f}")
    print(f"🟢 Avg Surplus Vol  : {avg_surplus_vol:,.2f}")
    
    return metrics

# ============= MAIN PIPELINE =============
# ============= WALK-FORWARD PIPELINE =============
def run_walk_forward_pipeline(residual_model='xgboost', ridge_alpha=2.0, n_trials=20, seed=42):
    """
    Simulates Real-Time Production Environment.
    Retrains model and recalibrates safety layer every month.
    """
    set_seed(seed)
    
    print("\n" + "="*60)
    print("🔄 STARTING ADAPTIVE WALK-FORWARD VALIDATION")
    print("="*60)

    # 1. Full Data Load
    df = load_and_process_data(FILE_PATH)
    if df is None: return None
    df_features = create_features(df)
    
    # Sort by date
    df_features = df_features.sort_values('Date').reset_index(drop=True)
    dates = df_features['Date'].unique()
    
    # Define Walk-Forward Split
    # Start simulating from the last 20% of timeline (equivalent to Test set)
    n_test = int(len(dates) * TEST_SIZE) 
    start_sim_date = dates[-n_test]
    
    print(f"📅 Simulation Period: {pd.to_datetime(start_sim_date).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")
    
    # Label Encoder (Fit on full history first to ensure consistency, or fit incrementally)
    # Use Robust Encoder for Production Safety
    le = RobustLabelEncoder()
    df_features['PWSID_enc'] = le.fit_transform(df_features['PWSID'])
    
    adapter = DriftAdapter(decay_rate=0.99)
    features = ['PWSID_enc', 'Month', 'Year', 'Is_Summer_Peak', 'lag_1', 'lag_12', 'diff_12', 'Temp_lag_1', 'Precip_lag_1']
    
    all_predictions = []
    
    # Store Config History for Reuse
    current_safety_cfg = None
    
    # Loop through each simulation step (Monthly)
    sim_dates = [d for d in dates if d >= start_sim_date]
    
    for i, current_date in enumerate(sim_dates):
        curr_ts = pd.Timestamp(current_date)
        print(f"\n📍 Forecasting Month: {curr_ts.strftime('%Y-%m')}")
        
        # 1. Define Dynamic Training Window (Expanding Window)
        train_mask = df_features['Date'] < current_date
        train_df = df_features[train_mask].copy()
        
        test_mask = df_features['Date'] == current_date
        test_df = df_features[test_mask].copy()
        
        if len(train_df) < 100: continue
        if test_df.empty: continue

        # 2. Adaptive Weights (Time-Decay)
        weights = adapter.calculate_sample_weights(train_df)
        
        # 3. Dynamic Validation Split
        # We take the LAST 12 months as our 'validation' set for monitoring & tuning
        val_start_dyn = curr_ts - pd.DateOffset(months=12)
        val_mask = train_df['Date'] >= val_start_dyn
        
        X_t_inner = train_df.loc[~val_mask, features]
        y_t_inner = train_df.loc[~val_mask, 'Demand']
        w_t_inner = weights[~val_mask]
        
        X_v_inner = train_df.loc[val_mask, features]
        y_v_inner = train_df.loc[val_mask, 'Demand']
        
        # 4. Always Retrain Base Model (It's fast enough and crucial for weights)
        model = HybridAnchorResidualModel(ridge_alpha=ridge_alpha, residual_model=residual_model, seed=seed)
        model.fit(X_t_inner, y_t_inner, sample_weight=w_t_inner, eval_set=[(X_v_inner, y_v_inner)])
        
        # 5. Smart Safety Calibration
        # Get recent predictions to check drift
        pred_val_inner = model.predict(X_v_inner)
        
        # Check Drift on last 3 months only (Immediate Drift)
        last_3m_mask = train_df.loc[val_mask, 'Date'] >= (curr_ts - pd.DateOffset(months=3))
        
        needs_retuning = False
        if last_3m_mask.sum() > 0:
             y_true_3m = y_v_inner[last_3m_mask]
             y_pred_3m = pred_val_inner[last_3m_mask]
             
             # NEW: Dynamic Thresholds based on recent mean demand
             # 15% MAE and 5% Shortage allowed (Adjustable)
             current_mean = np.mean(y_true_3m)
             th_mae = current_mean * 0.15
             th_shortage = current_mean * 0.02 # Strict on shortage
             
             drift_detected, metrics = adapter.monitor_drift(
                 y_true_3m, y_pred_3m, 
                 mae_threshold=th_mae, 
                 shortage_threshold=th_shortage
             )
             
             if drift_detected:
                 needs_retuning = True
        
        # Also force retune quarterly (Jan, Apr, Jul, Oct) to stay fresh
        if curr_ts.month in [1, 4, 7, 10]:
            print("   📅 Quarterly Maintenance: Force Retuning.")
            needs_retuning = True
            
        if current_safety_cfg is None: # First run
            needs_retuning = True
            
        # Optimize OR Reuse
        if needs_retuning:
            current_safety_cfg = optimize_safety_optuna(
                train_df[val_mask], y_v_inner, pred_val_inner, 
                n_trials=15, # Use sufficient trials
                seed=seed
            )
        else:
            print("   ⏩ Reuse Existing Safety Config (Save Compute).")
            
        safety_layer = WaterDemandSafetyLayer(current_safety_cfg)
        safety_layer.fit(train_df[val_mask], y_v_inner, pred_val_inner)
        
        # 6. Predict Next Month
        X_test = test_df[features]
        raw_pred = model.predict(X_test)
        final_pred = safety_layer.predict(raw_pred, test_df)
        
        test_df['Pred_Raw'] = raw_pred
        test_df['Pred_Final'] = final_pred
        all_predictions.append(test_df)

    # ============= AGGREGATE RESULTS =============
    if not all_predictions:
        print("❌ No predictions made.")
        return None
        
    full_res = pd.concat(all_predictions)
    
    print("\n" + "="*60)
    print("🏆 WALK-FORWARD ADAPTIVE RESULTS")
    print("="*60)
    
    eval_metrics = detailed_evaluation(full_res['Demand'], full_res['Pred_Final'], "ADAPTIVE_ROLLING_TEST")
    
    return full_res

if __name__ == "__main__":
    # Chạy mô phỏng Continuous Learning
    run_walk_forward_pipeline(residual_model='xgboost', ridge_alpha=2.0, n_trials=15, seed=42)
