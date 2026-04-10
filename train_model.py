"""
train_model.py
==============
Run this ONCE before launching the dashboard.
Command:  python train_model.py

What it does:
  1. Loads your CSV from the data/ folder
  2. Engineers all 87 features (congestion, weather, route risk, lag/rolling)
  3. Trains HistGradientBoosting + Random Forest
  4. Computes permutation importance (XAI)
  5. Saves everything to models/logistics_delay_model.pkl
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import logging

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier, IsolationForest)
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import permutation_importance

# ── folders ───────────────────────────────────────────────────────────────────
os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_PATH  = os.path.join("data", "logistics_disruption_dataset.csv")
MODEL_PATH = os.path.join("models", "logistics_delay_model.pkl")

# ── port GPS coords ────────────────────────────────────────────────────────────
PORT_COORDS = {
    "Colombo":(6.9447,79.8349),"Rotterdam":(51.9225,4.4791),"Dubai":(25.0325,55.0516),
    "Shanghai":(31.2165,121.4365),"Singapore":(1.293,103.8558),"Felixstowe":(51.961,1.351),
    "Hamburg":(53.55,10.0),"Los Angeles":(34.1139,-118.4068),"Busan":(35.1796,129.075),
    "Mumbai":(19.0758,72.8775),"Houston":(29.7868,-95.39),"Piraeus":(37.9431,23.6469),
    "Tokyo":(35.685,139.751),"Genoa":(44.4093,8.9328),"Antwerp":(51.2194,4.402),
    "Jeddah":(21.4858,39.1925),"Dar es Salaam":(-6.8,39.283),
    "Sydney":(-33.92,151.1852),"New York":(40.7,-74.016),"Vancouver":(49.2827,-123.1207),
}

ANOMALY_FEATURES = [
    "vessels_at_anchorage_count","berth_occupancy_pct","yard_utilization_pct",
    "median_waiting_time_hours","wind_speed_knots","wave_height_meters",
    "geopolitical_risk_index","strike_alert_level","port_congestion_index",
    "weather_severity_score","route_risk_score",
]
LAG_COLS = [
    "berth_occupancy_pct","yard_utilization_pct","vessels_at_anchorage_count",
    "median_waiting_time_hours","port_congestion_index","route_risk_score","weather_severity_score",
]
BASE_FEATURES = [
    "origin_port_enc","destination_port_enc","transport_mode_enc","carrier_id_enc",
    "scheduled_transit_days","distance_km","transit_efficiency","crossing_equator",
    "transit_delay_8h_bin","port_congestion_index","congestion_x_vessels","berth_over_yard",
    "high_berth","high_yard","critical_congestion","weather_severity_score","wind_x_wave",
    "fog_x_visibility","severe_weather","route_risk_score","geopolit_x_strike",
    "combined_disruption","fuel_x_risk","vessels_at_anchorage_count","median_waiting_time_hours",
    "berth_occupancy_pct","yard_utilization_pct","regional_fuel_price_index",
    "geopolitical_risk_index","news_sentiment_score","labor_strike_indicator",
    "port_closure_flag","strike_alert_level","wind_speed_knots","wave_height_meters",
    "visibility_km","fog_density_index","air_temperature_c","precipitation_mm",
    "anomaly_score","is_anomaly","hour","day_of_week","month","quarter",
    "is_weekend","is_peak_season","is_monsoon","orig_delay_hist","dest_delay_hist",
    "carrier_delay_hist","route_delay_hist","carrier_route_hist",
    "port_delay_30d","port_delay_7d","carrier_delay_30d","delay_lag1","delay_lag3","delay_lag7",
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(np.radians(lat2-lat1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


def load_and_clean(path):
    log.info(f"Loading {path} ...")
    df = pd.read_csv(path)
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["is_delayed"] = df["is_delayed"].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0}).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["strike_alert_level","geopolitical_risk_index","regional_fuel_price_index"]:
        mn,mx = df[c].min(), df[c].max()
        df[f"{c}_norm"] = (df[c]-mn)/(mx-mn+1e-9)
    log.info(f"Loaded {df.shape[0]:,} rows")
    return df


def engineer(df):
    df = df.copy()
    # geo
    df["origin_lat"]  = df["origin_port"].map(lambda x: PORT_COORDS.get(x,(np.nan,np.nan))[0])
    df["origin_lon"]  = df["origin_port"].map(lambda x: PORT_COORDS.get(x,(np.nan,np.nan))[1])
    df["dest_lat"]    = df["destination_port"].map(lambda x: PORT_COORDS.get(x,(np.nan,np.nan))[0])
    df["dest_lon"]    = df["destination_port"].map(lambda x: PORT_COORDS.get(x,(np.nan,np.nan))[1])
    df["distance_km"] = haversine(df["origin_lat"].fillna(0),df["origin_lon"].fillna(0),df["dest_lat"].fillna(0),df["dest_lon"].fillna(0))
    df["crossing_equator"]   = ((df["origin_lat"]*df["dest_lat"])<0).astype(int)
    df["transit_efficiency"] = df["scheduled_transit_days"]/(df["distance_km"]/1000+1)
    # congestion
    df["port_congestion_index"] = (df["berth_occupancy_pct"]+df["yard_utilization_pct"])/2
    df["congestion_x_vessels"]  = df["port_congestion_index"]*df["vessels_at_anchorage_count"]
    df["berth_over_yard"]       = df["berth_occupancy_pct"]/(df["yard_utilization_pct"]+1)
    df["high_berth"]            = (df["berth_occupancy_pct"]>75).astype(int)
    df["high_yard"]             = (df["yard_utilization_pct"]>75).astype(int)
    df["critical_congestion"]   = (df["port_congestion_index"]>80).astype(int)
    # weather
    df["weather_severity_score"] = ((df["wave_height_meters"]/(df["wave_height_meters"].max()+1e-9))+(df["wind_speed_knots"]/(df["wind_speed_knots"].max()+1e-9))+(1-df["visibility_km"]/(df["visibility_km"].max()+1e-9)))/3
    df["wind_x_wave"]      = df["wind_speed_knots"]*df["wave_height_meters"]
    df["fog_x_visibility"] = df["fog_density_index"]*(1/(df["visibility_km"]+0.1))
    df["severe_weather"]   = (df["weather_severity_score"]>0.55).astype(int)
    # route risk
    df["route_risk_score"]   = df["strike_alert_level"]*0.30+df["geopolitical_risk_index"]*0.30+(1-df["news_sentiment_score"].clip(-1,1))*0.15+df["labor_strike_indicator"]*0.15+df["port_closure_flag"]*0.10
    df["geopolit_x_strike"]  = df["geopolitical_risk_index"]*df["strike_alert_level"]
    df["combined_disruption"] = df["labor_strike_indicator"]+df["port_closure_flag"]+(df["strike_alert_level"]>1).astype(int)
    df["fuel_x_risk"]        = df["regional_fuel_price_index"]*df["route_risk_score"]
    # AIS bin
    df["transit_delay_days_raw"] = df["actual_transit_days"]-df["scheduled_transit_days"]
    df["transit_delay_8h_bin"]   = (df["transit_delay_days_raw"]*3).round(0)/3
    # anomaly
    scaler = StandardScaler()
    Xa = scaler.fit_transform(df[ANOMALY_FEATURES])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(Xa)
    df["anomaly_flag"]  = iso.predict(Xa)
    df["anomaly_score"] = -iso.score_samples(Xa)
    df["is_anomaly"]    = (df["anomaly_flag"]==-1).astype(int)
    # temporal
    df["hour"]=df["timestamp"].dt.hour; df["day_of_week"]=df["timestamp"].dt.dayofweek
    df["month"]=df["timestamp"].dt.month; df["quarter"]=df["timestamp"].dt.quarter
    df["is_weekend"]=(df["day_of_week"]>=5).astype(int)
    df["is_peak_season"]=df["month"].isin([11,12]).astype(int)
    df["is_monsoon"]=df["month"].isin([6,7,8,9]).astype(int)
    # historical delay rates
    for grp,name in [("origin_port","orig_delay_hist"),("destination_port","dest_delay_hist"),
                     ("carrier_id","carrier_delay_hist"),(["origin_port","destination_port"],"route_delay_hist"),
                     (["carrier_id","origin_port"],"carrier_route_hist")]:
        df[name]=df.groupby(grp)["is_delayed"].transform(lambda x: x.shift(1).expanding().mean())
    df["port_delay_30d"]=df.groupby("origin_port")["is_delayed"].transform(lambda x: x.shift(1).rolling(30,min_periods=5).mean())
    df["port_delay_7d"] =df.groupby("origin_port")["is_delayed"].transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df["carrier_delay_30d"]=df.groupby("carrier_id")["is_delayed"].transform(lambda x: x.shift(1).rolling(30,min_periods=5).mean())
    df["delay_lag1"]=df.groupby("origin_port")["is_delayed"].shift(1)
    df["delay_lag3"]=df.groupby("origin_port")["is_delayed"].shift(3)
    df["delay_lag7"]=df.groupby("origin_port")["is_delayed"].shift(7)
    # lag & rolling
    for col in LAG_COLS:
        df[f"{col}_lag1"] =df.groupby("origin_port")[col].shift(1)
        df[f"{col}_lag3"] =df.groupby("origin_port")[col].shift(3)
        df[f"{col}_roll7"]=df.groupby("origin_port")[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df[f"{col}_roll14"]=df.groupby("origin_port")[col].transform(lambda x: x.rolling(14,min_periods=3).mean())
    df=df.dropna().reset_index(drop=True)
    # encode
    encoders={}
    for col in ["origin_port","destination_port","transport_mode","carrier_id"]:
        le=LabelEncoder(); df[f"{col}_enc"]=le.fit_transform(df[col].astype(str)); encoders[col]=le
    lag_feats=[c for c in df.columns if any(s in c for s in ("_lag1","_lag3","_roll7","_roll14"))]
    features=list(dict.fromkeys(BASE_FEATURES+lag_feats))
    features=[f for f in features if f in df.columns]
    log.info(f"Engineered {df.shape[0]:,} rows × {len(features)} features")
    return df, features, scaler, iso, encoders


def main():
    log.info("="*60)
    log.info("  NEXUS LOGISTICS AI  —  MODEL TRAINING")
    log.info("="*60)

    df_raw = load_and_clean(DATA_PATH)
    df, features, scaler, iso, encoders = engineer(df_raw)
    X = df[features]; y = df["is_delayed"]

    split = int(len(X)*0.80)
    X_tr,X_te = X.iloc[:split],X.iloc[split:]
    y_tr,y_te = y.iloc[:split],y.iloc[split:]
    log.info(f"Train: {len(X_tr):,}   Test: {len(X_te):,}")

    log.info("Training HistGradientBoosting ...")
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=8, random_state=42)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:,1]; preds = model.predict(X_te)
    acc = accuracy_score(y_te,preds); auc = roc_auc_score(y_te,probs)
    log.info(f"Accuracy: {acc:.4f}   AUC: {auc:.4f}")

    log.info("Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)[:,1]; rf_auc = roc_auc_score(y_te,rf_probs)

    log.info("Cross-validation (3-fold) ...")
    tscv   = TimeSeriesSplit(n_splits=3)
    hgb_cv = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1).tolist()
    rf_cv = cross_val_score(rf, X, y, cv=tscv, scoring="accuracy", n_jobs=1).tolist()

    log.info("Permutation importance ...")
    idx  = X_te.sample(min(2000,len(X_te)), random_state=42).index
    perm = permutation_importance(model, X_te.loc[idx], y_te.loc[idx], n_repeats=5, scoring="accuracy", random_state=42, n_jobs=1)
    perm_df = pd.DataFrame({"feature":features,"importance":perm.importances_mean,"std":perm.importances_std}).sort_values("importance",ascending=False).reset_index(drop=True)

    fpr_t,tpr_t,_ = roc_curve(y_te,probs); fpr_r,tpr_r,_ = roc_curve(y_te,rf_probs)
    cm = confusion_matrix(y_te,preds).tolist()
    baseline_del = int(y_te.sum())
    rerouted     = probs > 0.60
    ai_del       = int((~rerouted&(y_te.values==1)).sum()+(rerouted&(y_te.values==1)).sum()*0.30)
    delay_red    = (baseline_del-ai_del)/max(baseline_del,1)*100

    bundle = {
        "model":model,"rf_model":rf,"features":features,"scaler":scaler,"iso":iso,"encoders":encoders,
        "tuned_acc":acc,"tuned_auc":auc,"rf_auc":round(rf_auc,4),"hgb_cv":hgb_cv,"rf_cv":rf_cv,
        "confusion_matrix":cm,
        "fpr_tuned":fpr_t[::max(1,len(fpr_t)//100)].tolist(),"tpr_tuned":tpr_t[::max(1,len(fpr_t)//100)].tolist(),
        "fpr_rf":fpr_r[::max(1,len(fpr_r)//100)].tolist(),"tpr_rf":tpr_r[::max(1,len(fpr_r)//100)].tolist(),
        "perm_importance":perm_df.to_dict(orient="records"),
        "baseline_delayed":baseline_del,"ai_delayed":ai_del,
        "delay_reduction_pct":round(delay_red,1),"cost_savings_k":round(delay_red/100*1200,1),
        "train_size":len(X_tr),"test_size":len(X_te),"n_features":len(features),
        "trained_at":pd.Timestamp.now().isoformat()[:19],
    }
    with open(MODEL_PATH,"wb") as f: pickle.dump(bundle,f)

    log.info("="*60)
    log.info(f"  Saved  →  {MODEL_PATH}")
    log.info(f"  Accuracy : {acc:.4f}   AUC : {auc:.4f}")
    log.info(f"  Features : {len(features)}")
    log.info(f"  Done!  Now run:  streamlit run app.py")
    log.info("="*60)


if __name__ == "__main__":
    main()
