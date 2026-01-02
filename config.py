"""
Global configuration file
"""

# Data paths
FILE_PATH = 'CaRDS.csv'

# Data splitting
TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_SEED = 42

# Feature list
FEATURES = [
    'PWSID_enc', 
    'Month', 
    'Year', 
    'Is_Summer_Peak', 
    'lag_1', 
    'lag_12', 
    'diff_12', 
    'CDD'
]

# Model parameters
MODEL_CONFIG = {
    'ridge_alpha': 2.0,
    'residual_model': 'xgboost',
    'n_trials': 50,
    'seed': 42
}

# Safety layer config template
SAFETY_CONFIG_TEMPLATE = {
    'summer_months': [6, 7, 8],
    'floor': {
        'enabled': True,
        'yoy_growth_min': 1.0,
        'mom_drop_max': 0.90,
        'mom_drop_summer': 1.0,
        'mom_drop_fall': 0.85
    },
    'buffer': {
        'enabled': True,
        'base_sigma': 2.0,
        'hist_coverage': 1.0,
        'summer_coverage': 1.5,
        'max_cap_pct': 0.3
    }
}