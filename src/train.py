import pandas as pd
import numpy as np
import os
import optuna
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler 
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from config import setup_logging, DATA_DIR, MODEL_DIR, set_seed
import joblib

logger = setup_logging(__name__)

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
        """Create residual model based on type with Fixed Seed"""
        if self.residual_model_name == 'xgboost':
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
            
        elif self.residual_model_name == 'lightgbm':
            default_params = {
                'objective': 'quantile',
                'alpha': 0.5,
                'n_estimators': 3000,
                'learning_rate': 0.02,
                'num_leaves': 31,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.seed,
                'n_jobs': -1
            }
            default_params.update(self.residual_params)
            return lgb.LGBMRegressor(**default_params)
            
        elif self.residual_model_name == 'catboost':
            default_params = {
                'loss_function': 'Quantile:alpha=0.5',
                'iterations': 3000,
                'learning_rate': 0.02,
                'depth': 6,
                'l2_leaf_reg': 1.0,
                'early_stopping_rounds': 200,
                'random_state': self.seed,
                'verbose': False
            }
            default_params.update(self.residual_params)
            return CatBoostRegressor(**default_params)
            
        elif self.residual_model_name == 'random_forest':
            default_params = {
                'n_estimators': 300,
                'max_depth': 12,
                'min_samples_split': 5,
                'random_state': self.seed,
                'n_jobs': -1
            }
            default_params.update(self.residual_params)
            return RandomForestRegressor(**default_params)
            
        else:
            raise ValueError(f"Unknown residual model: {self.residual_model_name}")
    
    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=False):
        """Two-stage training"""
        logger.info(f"   ⚓ Fitting Anchor Ridge Layer...")
        
        # Stage 1: Ridge
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y, sample_weight=sample_weight)
        
        y_pred_ridge = self.ridge.predict(X_scaled)
        residuals = y - y_pred_ridge
        
        ridge_mae = mean_absolute_error(y, y_pred_ridge)
        logger.info(f"      Ridge MAE: {ridge_mae:,.0f}")
        mlflow.log_metric("ridge_mae", ridge_mae)
        
        # Stage 2: Tree on residuals
        logger.info(f"   🚀 Fitting {self.residual_model_name.upper()} on Residuals...")
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
        
        # Fit based on model type
        if self.residual_model_name == 'xgboost':
            self.residual_model.fit(
                X, residuals,
                sample_weight=sample_weight,
                eval_set=eval_set_residual,
                verbose=verbose
            )
        elif self.residual_model_name == 'lightgbm':
            self.residual_model.fit(
                X, residuals,
                sample_weight=sample_weight,
                eval_set=eval_set_residual,
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
        elif self.residual_model_name == 'catboost':
            self.residual_model.fit(
                X, residuals,
                sample_weight=sample_weight,
                eval_set=eval_set_residual[0] if eval_set_residual else None
            )
        else:  # Random Forest
            self.residual_model.fit(X, residuals, sample_weight=sample_weight)
        
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

# ============= SAFETY OPTIMIZATION =============
def optimize_safety_optuna(val_df, y_val, raw_pred_val, n_trials=50, seed=42):
    """Optuna optimization with FIXED SEED"""
    logger.info("="*70)
    logger.info(f"🚀 STARTING OPTUNA OPTIMIZATION ({n_trials} trials) - STRICT MODE | SEED={seed}")
    logger.info("="*70)
    
    mean_demand = np.mean(y_val)
    DYNAMIC_S_MAX = mean_demand * 0.005
    target_under_rate = 0.02
    
    logger.info(f" ℹ️ Auto-tuning S_MAX constraint to: {DYNAMIC_S_MAX:,.0f}")
    
    val_df_sim = val_df.copy()
    val_df_sim['lag_1'] = np.nan
    
    def objective(trial):
        cfg = {
            'summer_months': [6, 7, 8],
            'floor': {
                'enabled': True,
                'yoy_growth_min': trial.suggest_float('yoy_min', 1.0, 1.25),
                'mom_drop_max': trial.suggest_float('mom_max', 0.85, 0.99),
                'mom_drop_summer': trial.suggest_float('mom_summer', 0.95, 1.05),
                'mom_drop_fall': trial.suggest_float('mom_fall', 0.75, 0.95)
            },
            'buffer': {
                'enabled': True,
                'base_sigma': trial.suggest_float('base_sigma', 1.0, 3.0),
                'hist_coverage': trial.suggest_float('hist_cov', 0.8, 1.5),
                'summer_coverage': trial.suggest_float('summer_cov', 1.0, 2.0),
                'max_cap_pct': trial.suggest_float('max_cap', 0.15, 0.5)
            }
        }
        
        layer = WaterDemandSafetyLayer(cfg)
        layer.fit(val_df_sim, y_val, raw_pred_val)
        preds = layer.predict(raw_pred_val, val_df_sim)
        
        diff = preds - y_val
        surplus_score = np.mean(np.maximum(diff, 0)) / mean_demand
        u_rate = np.mean(diff < 0)
        s_vol = np.mean(np.maximum(-diff, 0))
        
        penalty = 0
        if u_rate > target_under_rate:
            penalty += (u_rate - target_under_rate) * 5000
        if s_vol > DYNAMIC_S_MAX:
            penalty += (s_vol - DYNAMIC_S_MAX) / mean_demand * 200
        
        return surplus_score + penalty
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
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
    
    logger.info(f"✅ Optimization complete! Best score: {study.best_value:.6f}")
    return final_config

def main():
    set_seed()
    mlflow.set_experiment("Water_Demand_Forecast_Hybrid")
    
    logger.info("Loading train/val data...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val.parquet'))
    
    features = ['PWSID_enc', 'Month', 'Year', 'Is_Summer_Peak', 'lag_1', 'lag_12', 'diff_12', 'CDD']
    X_train, y_train = train_df[features], train_df['Demand']
    X_val, y_val = val_df[features], val_df['Demand']
    
    with mlflow.start_run() as run:
        # Params
        residual_model = 'xgboost'
        ridge_alpha = 2.0
        n_trials = 20 # Lowered from 50 for speed in demo
        
        mlflow.log_param("residual_model", residual_model)
        mlflow.log_param("ridge_alpha", ridge_alpha)
        mlflow.log_param("n_trials", n_trials)
        
        # 1. Train Hybrid Model
        logger.info(f"🚀 TRAINING {residual_model.upper()} HYBRID...")
        model = HybridAnchorResidualModel(ridge_alpha=ridge_alpha, residual_model=residual_model, seed=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Log model (using sklearn flavor for simplicity around custom model, or generic pyfunc)
        # For simplicity, saving as pickle then logging artifact
        joblib.dump(model, "model_hybrid.pkl")
        mlflow.log_artifact("model_hybrid.pkl")
        
        raw_pred_val = model.predict(X_val)
        
        # 2. Safety Optimization
        cfg = optimize_safety_optuna(val_df, y_val, raw_pred_val, n_trials=n_trials, seed=42)
        mlflow.log_params(cfg['floor'])
        
        final_layer = WaterDemandSafetyLayer(cfg)
        final_layer.fit(val_df, y_val, raw_pred_val)
        
        # Save Safety Layer
        joblib.dump(final_layer, "safety_layer.pkl")
        mlflow.log_artifact("safety_layer.pkl")
        
        # Save final Feature Config
        joblib.dump(features, "features_list.pkl")
        mlflow.log_artifact("features_list.pkl")
        
        logger.info(f"Training Complete. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
