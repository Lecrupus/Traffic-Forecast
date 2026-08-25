"""
Traffic Demand Prediction - Beat 93.12 Solution
Author: Intergalactic Hackathon God
Strategy:
  - 88.9% of test rows have exact (geohash, timestamp) match in day48 train data
  - Use day49 early-morning data (0:00-2:00) to compute per-geohash demand shift  
  - Key features: direct day48 lookup + full-train lookup + analytical offset prediction
  - LightGBM trained on full train with these analytical features
  - Blend ML predictions with pure additive-offset predictions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & PARSE
# ─────────────────────────────────────────────
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

def parse(df):
    df = df.copy()
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = ts[0]; df['minute'] = ts[1]
    df['time_min'] = ts[0]*60 + ts[1]
    df['geo_region'] = df['geohash'].str[:4]
    df['geo_sub']    = df['geohash'].str[:5]
    return df

train = parse(train); test = parse(test)
t48 = train[train['day']==48]; t49 = train[train['day']==49]

# ─────────────────────────────────────────────
# 2. COMPUTE DAY49 SHIFT (CRITICAL FEATURE)
# ─────────────────────────────────────────────
# Day49 train covers 0:00-2:00 (test covers 2:15-13:45)
# These early hours tell us how much demand shifted per geohash on day49
EARLY_TS = ['0:0','0:15','0:30','0:45','1:0','1:15','1:30','1:45','2:0']
geo_d48e  = t48[t48['timestamp'].isin(EARLY_TS)].groupby('geohash')['demand'].mean()
geo_d49e  = t49.groupby('geohash')['demand'].mean()

# Two adjustment strategies: additive offset & multiplicative scale
geo_off   = (geo_d49e - geo_d48e)                          # additive: better for absolute shift
geo_scale = (geo_d49e / (geo_d48e + 1e-8)).clip(0.05, 20) # multiplicative: better for relative shift
g_off     = float(geo_d49e.mean() - geo_d48e.mean())       # global fallback
g_scale   = float(geo_d49e.mean() / (geo_d48e.mean() + 1e-8))

# ─────────────────────────────────────────────
# 3. BUILD LOOKUP TABLES
# ─────────────────────────────────────────────
# Day48 lookups (the ground truth for day49 patterns)
t48_ex    = t48.set_index(['geohash','timestamp'])['demand'].to_dict()  # exact match
t48_ghtod = t48.groupby(['geohash','time_min'])['demand'].mean().to_dict()  # geo+time_of_day
t48_ghr   = t48.groupby(['geohash','hour'])['demand'].mean().to_dict()      # geo+hour
t48_gm    = t48.groupby('geohash')['demand'].mean().to_dict()               # geo mean
t48_tm    = t48.groupby('timestamp')['demand'].mean().to_dict()             # timestamp mean
t48_rm    = t48.groupby('geo_region')['demand'].mean().to_dict()            # region mean
t48_sm    = t48.groupby('geo_sub')['demand'].mean().to_dict()               # subregion mean
t48_std   = t48.groupby('geohash')['demand'].std().fillna(0).to_dict()
t48_g     = float(t48['demand'].mean())

# Full train lookups (88.9% of test keys exist in day48 = full train)
tr_ghtod  = train.groupby(['geohash','time_min'])['demand'].mean().to_dict()
tr_ghr    = train.groupby(['geohash','hour'])['demand'].mean().to_dict()
tr_gm     = train.groupby('geohash')['demand'].mean().to_dict()

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def fe(df):
    g  = df['geohash'].values; ts = df['timestamp'].values
    hr = df['hour'].values;    tm = df['time_min'].values
    gr = df['geo_region'].values; gs = df['geo_sub'].values
    temp = df['Temperature'].fillna(16.4).values
    nl   = df['NumberofLanes'].values

    # Multi-level day48 lookups (cascade from most to least specific)
    d48_ex    = np.array([t48_ex.get((g[i],ts[i]),    np.nan) for i in range(len(g))])
    d48_ghtod = np.array([t48_ghtod.get((g[i],tm[i]), np.nan) for i in range(len(g))])
    d48_ghr   = np.array([t48_ghr.get((g[i],hr[i]),   np.nan) for i in range(len(g))])
    d48_gm    = np.array([t48_gm.get(g[i],  np.nan) for i in range(len(g))])
    d48_tm    = np.array([t48_tm.get(ts[i], np.nan) for i in range(len(g))])
    d48_rm    = np.array([t48_rm.get(gr[i], np.nan) for i in range(len(g))])
    d48_sm    = np.array([t48_sm.get(gs[i], np.nan) for i in range(len(g))])
    d48_std   = np.array([t48_std.get(g[i], 0.0)    for i in range(len(g))])

    # Full train lookups
    tr_ghtod_v= np.array([tr_ghtod.get((g[i],tm[i]), np.nan) for i in range(len(g))])
    tr_ghr_v  = np.array([tr_ghr.get((g[i],hr[i]),   np.nan) for i in range(len(g))])
    tr_gm_v   = np.array([tr_gm.get(g[i],   np.nan)  for i in range(len(g))])

    # Day49 adjustment
    g_sc  = np.array([geo_scale.get(g[i], g_scale) for i in range(len(g))])
    g_of_v= np.array([geo_off.get(g[i],   g_off)   for i in range(len(g))])
    g_d49 = np.array([geo_d49e.get(g[i],  t48_g*g_scale) for i in range(len(g))])

    # Cascaded base from day48
    base = np.where(~np.isnan(d48_ex),    d48_ex,
           np.where(~np.isnan(d48_ghtod), d48_ghtod,
           np.where(~np.isnan(d48_ghr),   d48_ghr,
           np.where(~np.isnan(d48_gm),    d48_gm,
           np.where(~np.isnan(d48_tm),    d48_tm, t48_g)))))

    # KEY: Analytical predictions as features (tell ML what the answer should be)
    p_sc = np.clip(base * g_sc, 0, 1)   # scaled day48
    p_of = np.clip(base + g_of_v, 0, 1) # offset day48 (stronger signal)

    return pd.DataFrame({
        # Time features
        'hour': hr, 'minute': df['minute'].values, 'time_min': tm,
        'day': df['day'].values if 'day' in df.columns else np.full(len(g), 49),
        'hs': np.sin(2*np.pi*hr/24),   'hc': np.cos(2*np.pi*hr/24),
        'ms': np.sin(2*np.pi*tm/1440), 'mc': np.cos(2*np.pi*tm/1440),
        'im': ((hr>=7)&(hr<=9)).astype(int), 'ie': ((hr>=17)&(hr<=19)).astype(int),
        'in': ((hr>=22)|(hr<=5)).astype(int), 'id': ((hr>=11)&(hr<=13)).astype(int),
        # Road/weather
        'lv': (df['LargeVehicles']=='Allowed').astype(int).values,
        'lm': (df['Landmarks']=='Yes').astype(int).values,
        'rt': df['RoadType'].map({'Residential':0,'Street':1,'Highway':2}).fillna(-1).values,
        'wt': df['Weather'].map({'Sunny':0,'Foggy':1,'Rainy':2,'Snowy':3}).fillna(-1).values,
        'nl': nl, 'temp': temp, 't2': temp**2, 'txl': temp*nl,
        # Day48 lookups at multiple granularities
        'e48': d48_ex, 'g48': d48_ghtod, 'h48': d48_ghr, 'm48': d48_gm,
        'tm48': d48_tm, 'rm48': d48_rm, 'sm48': d48_sm, 'std48': d48_std, 'base': base,
        # Full train lookups (key: covers 88.9% of test)
        'trtod': tr_ghtod_v, 'trhr': tr_ghr_v, 'trgm': tr_gm_v,
        # Day49 adjustment values
        'gsc': g_sc, 'gof': g_of_v, 'gd49': g_d49,
        # Analytical predictions (THE KEY FEATURES)
        'psc': p_sc,    # day48 * geo_scale
        'pof': p_of,    # day48 + geo_offset   <-- strongest signal
        'hex': (~np.isnan(d48_ex)).astype(int),  # exact match flag
    })

print("Building features...")
Xtr = fe(train).fillna(0)
Xts = fe(test).fillna(0)
y   = train['demand'].values
FEATS = list(Xtr.columns)
print(f"Features: {len(FEATS)} | Train: {Xtr.shape} | Test: {Xts.shape}")

# ─────────────────────────────────────────────
# 5. LIGHTGBM
# ─────────────────────────────────────────────
print("Training LightGBM on full train...")
m = lgb.LGBMRegressor(
    objective='regression', n_estimators=2000, learning_rate=0.05,
    num_leaves=255, min_child_samples=10,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.05, reg_lambda=0.05, verbose=-1, n_jobs=-1)
m.fit(Xtr[FEATS], y)

tp_ml = np.clip(m.predict(Xts[FEATS]), 0, 1)
print(f"Train R²: {r2_score(y, np.clip(m.predict(Xtr[FEATS]),0,1)):.5f}")

# ─────────────────────────────────────────────
# 6. BLEND: ML + ANALYTICAL OFFSET
# ─────────────────────────────────────────────
p_of_test = Xts['pof'].values
# 70% ML (captures non-linear patterns) + 30% analytical offset (handles scale shift)
final = np.clip(0.7*tp_ml + 0.3*p_of_test, 0, 1)
print(f"\nTest prediction mean: {final.mean():.4f}")

# ─────────────────────────────────────────────
# 7. SUBMISSION
# ─────────────────────────────────────────────
sub = pd.DataFrame({'Index': test['Index'], 'demand': final})
sub.to_csv('submission_93plus_lgb_offset.csv', index=False)
print(f"submission_93plus_lgb_offset.csv saved! Shape: {sub.shape}")
print(sub.head())
