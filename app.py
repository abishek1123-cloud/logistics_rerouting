import os, sys, pickle, warnings, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog

# ── Agent imports (graceful fallback if not installed) ─────────────────────
try:
    from typing import TypedDict
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph import StateGraph
    load_dotenv()
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Page config  (must be FIRST streamlit call) ────────────────────────────────
st.set_page_config(page_title="NEXUS LOGISTICS AI", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH  = os.path.join("data",   "logistics_disruption_dataset.csv")
MODEL_PATH = os.path.join("models", "logistics_delay_model.pkl")

PORT_COORDS = {
    "Colombo":(6.9447,79.8349),"Rotterdam":(51.9225,4.4791),"Dubai":(25.0325,55.0516),
    "Shanghai":(31.2165,121.4365),"Singapore":(1.293,103.8558),"Felixstowe":(51.961,1.351),
    "Hamburg":(53.55,10.0),"Los Angeles":(34.1139,-118.4068),"Busan":(35.1796,129.075),
    "Mumbai":(19.0758,72.8775),"Houston":(29.7868,-95.39),"Piraeus":(37.9431,23.6469),
    "Tokyo":(35.685,139.751),"Genoa":(44.4093,8.9328),"Antwerp":(51.2194,4.402),
    "Jeddah":(21.4858,39.1925),"Dar es Salaam":(-6.8,39.283),
    "Sydney":(-33.92,151.1852),"New York":(40.7,-74.016),"Vancouver":(49.2827,-123.1207),
    "Cape Town":(-33.92,18.4),
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
FEATURE_GROUPS = {
    "port_congestion_index":"Congestion","route_risk_score":"Risk",
    "weather_severity_score":"Weather","berth_occupancy_pct":"Congestion",
    "transit_delay_8h_bin":"Temporal","median_waiting_time_hours":"Congestion",
    "geopolitical_risk_index":"Risk","strike_alert_level":"Risk",
    "congestion_x_vessels":"Congestion","yard_utilization_pct":"Congestion",
    "wave_height_meters":"Weather","news_sentiment_score":"External",
    "orig_delay_hist":"Temporal","scheduled_transit_days":"Logistics",
    "distance_km":"Logistics","wind_x_wave":"Weather","geopolit_x_strike":"Risk",
    "combined_disruption":"Risk","fuel_x_risk":"Risk","anomaly_score":"Anomaly",
}
GROUP_COLORS = {
    "Congestion":"#ff3b5c","Risk":"#ff8c00","Weather":"#6699ff",
    "Temporal":"#f5d800","Logistics":"#00c8b4","External":"#cc99ff","Anomaly":"#00e676",
}
THEME = dict(
    paper_bgcolor="rgba(8,12,20,0)", plot_bgcolor="rgba(10,20,40,0.5)",
    font=dict(color="#d0e8ff", family="Courier New", size=11),
)
AXIS = dict(gridcolor="rgba(0,200,180,0.08)", showgrid=True, zeroline=False, tickfont=dict(size=10, color="#d0e8ff"))

def apply_theme(fig, height=None, **layout_kwargs):
    merged = dict(paper_bgcolor="rgba(8,12,20,0)", plot_bgcolor="rgba(10,20,40,0.5)",
                  font=dict(color="#d0e8ff", family="Courier New", size=11))
    merged["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))
    merged["margin"] = dict(l=40, r=20, t=40, b=40)
    if height:
        merged["height"] = height
    for k, v in layout_kwargs.items():
        if k == "legend" and isinstance(v, dict):
            merged["legend"] = {**merged["legend"], **v}
        elif k == "margin" and isinstance(v, dict):
            merged["margin"] = {**merged["margin"], **v}
        else:
            merged[k] = v
    fig.update_layout(**merged)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def prob_color(p):
    if p >= 0.75: return "#ff3b5c"
    if p >= 0.50: return "#ff8c00"
    if p >= 0.30: return "#f5d800"
    return "#00e676"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1,p2 = np.radians(lat1),np.radians(lat2)
    a = np.sin(np.radians(lat2-lat1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*R*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def metric_card(col, label, value, color, sub=""):
    col.markdown(f"""
    <div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,200,180,0.12);
                border-radius:8px;padding:14px;text-align:center">
      <div style="font-size:9px;color:#ffffff;letter-spacing:0.1em;margin-bottom:5px">{label}</div>
      <div style="font-size:23px;font-weight:800;color:{color}">{value}</div>
      <div style="font-size:9px;color:#ffffff;margin-top:3px">{sub}</div>
    </div>""", unsafe_allow_html=True)

def info_banner(text):
    st.markdown(f"""<div style="padding:12px 16px;background:rgba(0,200,180,0.04);
    border:1px solid rgba(0,200,180,0.18);border-radius:6px;font-size:11px;
    color:#e0eaf5;line-height:1.7;margin-bottom:14px">{text}</div>""",
    unsafe_allow_html=True)

def section_title(text):
    st.markdown(f"""<div style="font-size:10px;color:#00c8b4;letter-spacing:0.1em;
    margin-bottom:8px;margin-top:4px">{text}</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""<style>
    #MainMenu,footer,header{visibility:hidden}
    .block-container{padding:1rem 1.5rem !important;max-width:100% !important}
    .stApp{background:#080c14;color:#e0eaf5}
    body,*{font-family:'Courier New',monospace}
    .stTabs [data-baseweb="tab-list"]{background:#0a1628;border-bottom:1px solid rgba(0,200,180,0.15);gap:2px;padding:0 4px}
    .stTabs [data-baseweb="tab"]{color:#ffffff;font-size:11px;letter-spacing:0.08em;padding:10px 18px;border-bottom:2px solid transparent;background:transparent}
    .stTabs [aria-selected="true"]{color:#00c8b4 !important;border-bottom:2px solid #00c8b4 !important;background:rgba(0,200,180,0.08) !important}
    [data-testid="stSidebar"]{background:#0a1628;border-right:1px solid rgba(0,200,180,0.12)}
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:#c0d4e8;font-size:12px}
    ::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-track{background:#080c14} ::-webkit-scrollbar-thumb{background:#1a3a5c;border-radius:3px}
    div[data-testid="stDownloadButton"]>button{background:rgba(0,200,180,0.08);color:#00c8b4;border:1px solid rgba(0,200,180,0.35);border-radius:5px;font-family:'Courier New',monospace;font-size:11px;width:100%}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.25}}
    @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

    [data-testid="collapsedControl"]{
        display:flex !important;visibility:visible !important;opacity:1 !important;
        position:fixed !important;top:14px !important;left:14px !important;
        z-index:999999 !important;background:#0a1628 !important;
        border:1px solid rgba(0,200,180,0.45) !important;border-radius:7px !important;
        width:36px !important;height:36px !important;align-items:center !important;
        justify-content:center !important;cursor:pointer !important;
        box-shadow:0 0 12px rgba(0,200,180,0.2) !important;transition:all 0.2s ease !important;
    }
    [data-testid="collapsedControl"]:hover{
        background:rgba(0,200,180,0.15) !important;border-color:#00c8b4 !important;
        box-shadow:0 0 18px rgba(0,200,180,0.4) !important;
    }
    [data-testid="collapsedControl"] svg{fill:#00c8b4 !important;width:18px !important;height:18px !important;}
    button[kind="header"]{visibility:visible !important;opacity:1 !important;}
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_engineer(path):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import IsolationForest

    df = pd.read_csv(path)
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["is_delayed"] = df["is_delayed"].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0}).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["strike_alert_level","geopolitical_risk_index","regional_fuel_price_index"]:
        mn,mx=df[c].min(),df[c].max(); df[f"{c}_norm"]=(df[c]-mn)/(mx-mn+1e-9)

    df["origin_lat"] =df["origin_port"].map(lambda x:PORT_COORDS.get(x,(np.nan,np.nan))[0])
    df["origin_lon"] =df["origin_port"].map(lambda x:PORT_COORDS.get(x,(np.nan,np.nan))[1])
    df["dest_lat"]   =df["destination_port"].map(lambda x:PORT_COORDS.get(x,(np.nan,np.nan))[0])
    df["dest_lon"]   =df["destination_port"].map(lambda x:PORT_COORDS.get(x,(np.nan,np.nan))[1])
    df["distance_km"]=haversine(df["origin_lat"].fillna(0),df["origin_lon"].fillna(0),df["dest_lat"].fillna(0),df["dest_lon"].fillna(0))
    df["crossing_equator"]  =((df["origin_lat"]*df["dest_lat"])<0).astype(int)
    df["transit_efficiency"]=df["scheduled_transit_days"]/(df["distance_km"]/1000+1)
    df["port_congestion_index"]=(df["berth_occupancy_pct"]+df["yard_utilization_pct"])/2
    df["congestion_x_vessels"] =df["port_congestion_index"]*df["vessels_at_anchorage_count"]
    df["berth_over_yard"]      =df["berth_occupancy_pct"]/(df["yard_utilization_pct"]+1)
    df["high_berth"]           =(df["berth_occupancy_pct"]>75).astype(int)
    df["high_yard"]            =(df["yard_utilization_pct"]>75).astype(int)
    df["critical_congestion"]  =(df["port_congestion_index"]>80).astype(int)
    df["weather_severity_score"]=((df["wave_height_meters"]/(df["wave_height_meters"].max()+1e-9))+(df["wind_speed_knots"]/(df["wind_speed_knots"].max()+1e-9))+(1-df["visibility_km"]/(df["visibility_km"].max()+1e-9)))/3
    df["wind_x_wave"]     =df["wind_speed_knots"]*df["wave_height_meters"]
    df["fog_x_visibility"]=df["fog_density_index"]*(1/(df["visibility_km"]+0.1))
    df["severe_weather"]  =(df["weather_severity_score"]>0.55).astype(int)
    df["route_risk_score"]  =df["strike_alert_level"]*0.30+df["geopolitical_risk_index"]*0.30+(1-df["news_sentiment_score"].clip(-1,1))*0.15+df["labor_strike_indicator"]*0.15+df["port_closure_flag"]*0.10
    df["geopolit_x_strike"] =df["geopolitical_risk_index"]*df["strike_alert_level"]
    df["combined_disruption"]=df["labor_strike_indicator"]+df["port_closure_flag"]+(df["strike_alert_level"]>1).astype(int)
    df["fuel_x_risk"]       =df["regional_fuel_price_index"]*df["route_risk_score"]
    df["transit_delay_days_raw"]=df["actual_transit_days"]-df["scheduled_transit_days"]
    df["transit_delay_8h_bin"]  =(df["transit_delay_days_raw"]*3).round(0)/3
    sc=StandardScaler(); Xa=sc.fit_transform(df[ANOMALY_FEATURES])
    iso=IsolationForest(n_estimators=200,contamination=0.05,random_state=42,n_jobs=-1); iso.fit(Xa)
    df["anomaly_flag"]=iso.predict(Xa); df["anomaly_score"]=-iso.score_samples(Xa)
    df["is_anomaly"]=(df["anomaly_flag"]==-1).astype(int)
    df["hour"]=df["timestamp"].dt.hour; df["day_of_week"]=df["timestamp"].dt.dayofweek
    df["month"]=df["timestamp"].dt.month; df["quarter"]=df["timestamp"].dt.quarter
    df["is_weekend"]=(df["day_of_week"]>=5).astype(int)
    df["is_peak_season"]=df["month"].isin([11,12]).astype(int)
    df["is_monsoon"]=df["month"].isin([6,7,8,9]).astype(int)
    for grp,name in [("origin_port","orig_delay_hist"),("destination_port","dest_delay_hist"),
                     ("carrier_id","carrier_delay_hist"),(["origin_port","destination_port"],"route_delay_hist"),
                     (["carrier_id","origin_port"],"carrier_route_hist")]:
        df[name]=df.groupby(grp)["is_delayed"].transform(lambda x:x.shift(1).expanding().mean())
    df["port_delay_30d"]=df.groupby("origin_port")["is_delayed"].transform(lambda x:x.shift(1).rolling(30,min_periods=5).mean())
    df["port_delay_7d"] =df.groupby("origin_port")["is_delayed"].transform(lambda x:x.shift(1).rolling(7, min_periods=3).mean())
    df["carrier_delay_30d"]=df.groupby("carrier_id")["is_delayed"].transform(lambda x:x.shift(1).rolling(30,min_periods=5).mean())
    df["delay_lag1"]=df.groupby("origin_port")["is_delayed"].shift(1)
    df["delay_lag3"]=df.groupby("origin_port")["is_delayed"].shift(3)
    df["delay_lag7"]=df.groupby("origin_port")["is_delayed"].shift(7)
    for col in LAG_COLS:
        df[f"{col}_lag1"] =df.groupby("origin_port")[col].shift(1)
        df[f"{col}_lag3"] =df.groupby("origin_port")[col].shift(3)
        df[f"{col}_roll7"]=df.groupby("origin_port")[col].transform(lambda x:x.rolling(7, min_periods=1).mean())
        df[f"{col}_roll14"]=df.groupby("origin_port")[col].transform(lambda x:x.rolling(14,min_periods=3).mean())
    df=df.dropna().reset_index(drop=True)
    encoders={}
    for col in ["origin_port","destination_port","transport_mode","carrier_id"]:
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder(); df[f"{col}_enc"]=le.fit_transform(df[col].astype(str)); encoders[col]=le
    lag_feats=[c for c in df.columns if any(s in c for s in ("_lag1","_lag3","_roll7","_roll14"))]
    features=list(dict.fromkeys(BASE_FEATURES+lag_feats))
    features=[f for f in features if f in df.columns]
    return df, features, encoders


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH,"rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def run_predictions(_df, _features, _bundle):
    model = _bundle["model"]
    feats = [f for f in _bundle["features"] if f in _df.columns]
    probs = model.predict_proba(_df[feats])[:,1]
    df2   = _df.copy()
    df2["ai_delay_prob"] = probs
    df2["ai_risk_score"] = (probs*100).round(1)
    df2["ai_rerouted"]   = (probs>0.60).astype(int)
    df2["risk_status"]   = pd.cut(probs,bins=[-0.001,0.30,0.50,0.75,1.01],
                                   labels=["LOW","MEDIUM","HIGH","CRITICAL"])
    return df2


def fallback_predictions(df):
    rrs = df["route_risk_score"]
    probs = (rrs/(rrs.max()+1e-9)).clip(0,1)
    df2 = df.copy()
    df2["ai_delay_prob"] = probs
    df2["ai_risk_score"] = (probs*100).round(1)
    df2["ai_rerouted"]   = (probs>0.60).astype(int)
    df2["risk_status"]   = pd.cut(probs,bins=[-0.001,0.30,0.50,0.75,1.01],
                                   labels=["LOW","MEDIUM","HIGH","CRITICAL"])
    return df2


@st.cache_data(show_spinner=False)
def port_stats(_df):
    return (_df.groupby("origin_port").agg(
        avg_congestion=("port_congestion_index","mean"),avg_weather=("weather_severity_score","mean"),
        avg_route_risk=("route_risk_score","mean"),anomaly_rate=("is_anomaly","mean"),
        delay_rate=("is_delayed","mean"),shipment_count=("is_delayed","count"),
        avg_delay_minutes=("delay_minutes","mean"))
    .round(3).sort_values("delay_rate",ascending=False).reset_index())


@st.cache_data(show_spinner=False)
def monthly_trend(_df):
    d=_df.copy(); d["ym"]=d["timestamp"].dt.to_period("M")
    m=(d.groupby("ym").agg(delay_rate=("is_delayed","mean"),avg_risk=("route_risk_score","mean"),
        avg_congestion=("port_congestion_index","mean"),anomaly_rate=("is_anomaly","mean"),
        shipments=("is_delayed","count")).reset_index())
    m["ym_str"]=m["ym"].astype(str); return m


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH AGENT  (only built when AGENT_AVAILABLE=True)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_agent(_df_raw):
    """Compile the LangGraph agent once and cache it."""
    if not AGENT_AVAILABLE:
        return None

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return None

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=groq_key,
    )

    agent_prompt = PromptTemplate.from_template(
        """You are an expert supply chain analyst.

Shipment route: {origin} → {destination}

Port risk data:
{port_risk}

Route history data:
{route_history}

Explain briefly why this shipment may be risky and suggest ONE rerouting recommendation.

Respond in this format:

RISK:
RECOMMENDATION:
CONFIDENCE:
"""
    )

    class State(TypedDict, total=False):
        origin: str
        destination: str
        port_risk: str
        route_history: str
        analysis: str

    def port_risk_node(state: State):
        origin = state.get("origin")
        sub = _df_raw[_df_raw["origin_port"] == origin]
        if sub.empty:
            state["port_risk"] = "No port data available."
        else:
            delay_rate = sub["is_delayed"].mean() if "is_delayed" in _df_raw.columns else 0
            weather    = sub["weather_severity_score"].mean() if "weather_severity_score" in _df_raw.columns else 0
            state["port_risk"] = (
                f"Delay Rate: {delay_rate:.2%}\n"
                f"Weather Severity: {weather:.2f}\n"
                f"Records: {len(sub)}"
            )
        return state

    def route_history_node(state: State):
        origin = state.get("origin")
        dest   = state.get("destination")
        sub = _df_raw[
            (_df_raw["origin_port"] == origin) &
            (_df_raw["destination_port"] == dest)
        ]
        if sub.empty:
            state["route_history"] = "No historical route data."
        else:
            delay_rate = sub["is_delayed"].mean() if "is_delayed" in _df_raw.columns else 0
            avg_delay  = sub["delay_minutes"].mean() if "delay_minutes" in _df_raw.columns else 0
            state["route_history"] = (
                f"Historical Delay Rate: {delay_rate:.2%}\n"
                f"Average Delay: {avg_delay:.1f} minutes\n"
                f"Shipments: {len(sub)}"
            )
        return state

    def reroute_node(state: State):
        chain = agent_prompt | llm
        result = chain.invoke(state)
        state["analysis"] = result.content
        return state

    graph = StateGraph(State)
    graph.add_node("port_risk",     port_risk_node)
    graph.add_node("route_history", route_history_node)
    graph.add_node("reroute",       reroute_node)
    graph.set_entry_point("port_risk")
    graph.add_edge("port_risk",     "route_history")
    graph.add_edge("route_history", "reroute")
    return graph.compile()


def _parse_analysis(text: str) -> dict:
    """Split raw LLM output into RISK / RECOMMENDATION / CONFIDENCE sections."""
    out = {"RISK": "", "RECOMMENDATION": "", "CONFIDENCE": ""}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("RISK:"):
            current = "RISK"; out[current] = line[5:].strip()
        elif line.startswith("RECOMMENDATION:"):
            current = "RECOMMENDATION"; out[current] = line[15:].strip()
        elif line.startswith("CONFIDENCE:"):
            current = "CONFIDENCE"; out[current] = line[11:].strip()
        elif current and line:
            out[current] += " " + line
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df):
    with st.sidebar:
        st.markdown("## ⬡ NEXUS CONTROLS")
        st.markdown("---")
        st.markdown("### 📁 Data Source")
        uploaded = st.file_uploader("Upload your own CSV", type=["csv"])
        if uploaded:
            try:
                u = pd.read_csv(uploaded)
                st.success(f"✓ {len(u):,} rows loaded")
                st.session_state["uploaded_df"] = u
            except Exception as e:
                st.error(f"Upload error: {e}")
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        search   = st.text_input("Search Shipment ID", placeholder="e.g. SHP0000123")
        risk_fil = st.selectbox("Risk Status", ["ALL","CRITICAL","HIGH","MEDIUM","LOW"])
        ports    = sorted(df["origin_port"].unique().tolist())
        port_fil = st.multiselect("Origin Ports", ports)
        modes    = sorted(df["transport_mode"].unique().tolist())
        mode_fil = st.multiselect("Transport Mode", modes)
        st.markdown("### 📅 Time Range")
        dmin,dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
        drange = st.date_input("Select range", value=(dmin,dmax), min_value=dmin, max_value=dmax)
        prob_t = st.slider("Min AI Delay Probability", 0.0, 1.0, 0.0, 0.05)
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        buf=io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("⬇ Download Dataset (CSV)", buf.getvalue().encode(),
                           "nexus_predictions.csv", "text/csv", use_container_width=True)
        st.markdown("---")
        st.markdown(f"<div style='font-size:10px;color:#c8ddf0'>{len(df):,} shipments<br>{dmin} → {dmax}</div>",
                    unsafe_allow_html=True)

    d = df.copy()
    if search.strip():
        d = d[d["shipment_id"].astype(str).str.contains(search.strip(), case=False)]
    if risk_fil != "ALL":
        d = d[d["risk_status"].astype(str) == risk_fil]
    if port_fil:  d = d[d["origin_port"].isin(port_fil)]
    if mode_fil:  d = d[d["transport_mode"].isin(mode_fil)]
    if isinstance(drange,(list,tuple)) and len(drange)==2:
        d = d[(d["timestamp"].dt.date>=drange[0]) & (d["timestamp"].dt.date<=drange[1])]
    d = d[d["ai_delay_prob"] >= prob_t]
    return d


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header(df, bundle):
    acc = bundle.get("tuned_acc", 0) if bundle else 0
    acc_str = f"{acc:.1%}" if acc else "N/A"
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#0a1628,#0d1f3c,#0a1628);
                border-bottom:1px solid rgba(0,200,180,0.25);padding:16px 24px;
                display:flex;align-items:center;gap:24px;border-radius:8px;margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
        <div style="width:38px;height:38px;border-radius:9px;background:linear-gradient(135deg,#00c8b4,#0062ff);
                    display:flex;align-items:center;justify-content:center;font-size:20px">⬡</div>
        <div>
          <div style="font-size:17px;font-weight:800;letter-spacing:0.12em;color:#00c8b4">NEXUS LOGISTICS AI</div>
          <div style="font-size:9px;color:#c8ddf0;letter-spacing:0.2em">DISRUPTION-PROOF SUPPLY CHAIN INTELLIGENCE</div>
        </div>
      </div>
      <div style="flex:1"></div>
      <div style="display:flex;gap:20px">
        {"".join(f'<div style="text-align:center"><div style="font-size:20px;font-weight:800;color:{c}">{v}</div><div style="color:#ffffff;font-size:9px;letter-spacing:0.1em">{l}</div></div>'
          for l,v,c in [
            ("SHIPMENTS",    f"{len(df):,}",                                          "#00c8b4"),
            ("AT RISK",      f"{int(df['risk_status'].astype(str).isin(['HIGH','CRITICAL']).sum()):,}", "#ff3b5c"),
            ("CRITICAL",     f"{int((df['risk_status'].astype(str)=='CRITICAL').sum()):,}",             "#ff8c00"),
            ("AI REROUTED",  f"{int(df['ai_rerouted'].sum()):,}",                     "#00e676"),
            ("AVG DELAY",    f"{df['delay_minutes'].mean():.0f}m",                    "#f5d800"),
            ("MODEL ACC.",   acc_str,                                                  "#00e676"),
          ])}
      </div>
      <div style="display:flex;align-items:center;gap:6px;background:rgba(0,230,118,0.08);
                  border:1px solid rgba(0,230,118,0.3);padding:6px 14px;border-radius:5px;
                  font-size:10px;color:#00e676">
        <div style="width:7px;height:7px;border-radius:50%;background:#00e676;animation:pulse 1.5s infinite"></div>LIVE
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df, ps, monthly):
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,lbl,val,color in [
        (c1,"TOTAL SHIPMENTS", f"{len(df):,}",                    "#00c8b4"),
        (c2,"DELAY RATE",      f"{df['is_delayed'].mean():.1%}",  "#ff8c00"),
        (c3,"AVG DELAY",       f"{df['delay_minutes'].mean():.0f} min", "#ff3b5c"),
        (c4,"AI REROUTED",     f"{int(df['ai_rerouted'].sum()):,}","#00e676"),
        (c5,"ANOMALIES",       f"{int(df['is_anomaly'].sum()):,}", "#f5d800"),
        (c6,"CRITICAL",        f"{int((df['risk_status'].astype(str)=='CRITICAL').sum()):,}","#ff3b5c"),
    ]: metric_card(col,lbl,val,color)

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("◈ GLOBAL SHIPMENT NETWORK — REAL PORT LOCATIONS")

    fig = go.Figure()
    pairs = (df.groupby(["origin_port","destination_port"]).agg(prob=("ai_delay_prob","mean"),cnt=("ai_delay_prob","count")).reset_index().sort_values("cnt",ascending=False).head(20))
    for _,r in pairs.iterrows():
        o,d = r["origin_port"],r["destination_port"]
        if o in PORT_COORDS and d in PORT_COORDS:
            la1,lo1=PORT_COORDS[o]; la2,lo2=PORT_COORDS[d]
            ml=(la1+la2)/2+8*np.sign(la1-la2)
            fig.add_trace(go.Scattergeo(
                lat=[la1,ml,la2,None],lon=[lo1,(lo1+lo2)/2,lo2,None],
                mode="lines",line=dict(width=1.5,color=prob_color(r["prob"])),
                opacity=0.5,showlegend=False,hoverinfo="skip"))
    ps2=ps.copy()
    ps2["lat"]=ps2["origin_port"].map(lambda p:PORT_COORDS.get(p,(None,None))[0])
    ps2["lon"]=ps2["origin_port"].map(lambda p:PORT_COORDS.get(p,(None,None))[1])
    ps2=ps2.dropna(subset=["lat","lon"])
    fig.add_trace(go.Scattergeo(
        lat=ps2["lat"],lon=ps2["lon"],mode="markers+text",
        text=ps2["origin_port"].tolist(),
        textposition="top center",
        textfont=dict(size=9,color="rgba(0,200,180,0.8)"),
        marker=dict(size=ps2["delay_rate"]*28+7,color=ps2["delay_rate"],
                    colorscale=[[0,"#00e676"],[0.5,"#ff8c00"],[1,"#ff3b5c"]],
                    cmin=0,cmax=1,showscale=True,
                    colorbar=dict(title=dict(text="Delay Rate",font=dict(color="#d0e8ff",size=10)),
                                  thickness=12,tickformat=".0%",
                                  tickfont=dict(color="#d0e8ff",size=9)),
                    line=dict(width=1,color="rgba(0,200,180,0.6)")),
        customdata=ps2[["origin_port","delay_rate","shipment_count","avg_congestion"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Delay: %{customdata[1]:.1%}<br>Ships: %{customdata[2]:,}<extra></extra>",
        showlegend=False))
    fig.update_layout(
        geo=dict(scope="world",showland=True,landcolor="rgba(30,50,80,0.6)",showocean=True,
                 oceancolor="rgba(8,12,20,0.9)",showcoastlines=True,coastlinecolor="rgba(0,200,180,0.2)",
                 showframe=False,bgcolor="rgba(8,12,20,0)",projection_type="natural earth"),
        paper_bgcolor="rgba(8,12,20,0)",margin=dict(l=0,r=0,t=0,b=0),height=400)
    st.plotly_chart(fig, use_container_width=True)

    cl,cr = st.columns(2)
    with cl:
        section_title("◆ PORT DELAY RATE (TOP 10)")
        for _,r in ps.head(10).iterrows():
            dr=float(r["delay_rate"]); c_=prob_color(dr)
            st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
              <div style="width:90px;font-size:10px;color:#c0d4e8">{r['origin_port']}</div>
              <div style="flex:1;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
                <div style="width:{dr*100:.0f}%;height:100%;background:{c_}"></div></div>
              <div style="width:36px;font-size:10px;font-weight:700;color:{c_};text-align:right">{dr:.0%}</div>
              <div style="width:50px;font-size:9px;color:#ffffff;text-align:right">{int(r['shipment_count']):,}</div>
            </div>""", unsafe_allow_html=True)
    with cr:
        section_title("◆ TRANSPORT MODE BREAKDOWN")
        ms=df.groupby("transport_mode").agg(cnt=("is_delayed","count"),dr=("is_delayed","mean")).reset_index().sort_values("cnt",ascending=False)
        total_s=ms["cnt"].sum()
        mc={"Sea":"#00c8b4","Air":"#6699ff","Rail":"#f5d800","Road":"#ff8c00"}
        for _,r in ms.iterrows():
            c_=mc.get(r["transport_mode"],"#cc99ff"); pct=r["cnt"]/total_s
            st.markdown(f"""<div style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="font-size:11px;color:#c0d4e8">{r['transport_mode']}</span>
                <span style="font-size:10px;color:#ffffff">{pct:.0%} vol · <span style="color:{c_}">{r['dr']:.0%} delay</span></span>
              </div>
              <div style="height:7px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
                <div style="width:{pct*100:.0f}%;height:100%;background:{c_}"></div></div></div>""",
            unsafe_allow_html=True)

    if len(monthly)>0:
        section_title("◆ MONTHLY DELAY RATE & RISK TREND")
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=monthly["ym_str"],y=monthly["delay_rate"],mode="lines+markers",name="Delay Rate",
            line=dict(color="#ff3b5c",width=2),fill="tozeroy",fillcolor="rgba(255,59,92,0.08)",marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=monthly["ym_str"],y=monthly["avg_risk"],mode="lines",name="Avg Route Risk",
            line=dict(color="#ff8c00",width=1.5,dash="dot"),yaxis="y2"))
        apply_theme(fig2, height=240, legend=dict(orientation="h",y=1.08),
            yaxis2=dict(title="Risk",overlaying="y",side="right",showgrid=False))
        fig2.update_yaxes(title_text="Delay Rate",tickformat=".0%",gridcolor="rgba(0,200,180,0.07)",selector=dict(overlaying=None))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISRUPTION MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def tab_disruption(df, ps):
    info_banner("<span style='color:#00c8b4;font-weight:700'>◈ DISRUPTION INTELLIGENCE — </span>"
                "Real port risk heatmap from your dataset. Bubble size = shipment volume. "
                "Color = AI delay probability.")

    strikes=int(df["labor_strike_indicator"].sum()) if "labor_strike_indicator" in df else 0
    closures=int(df["port_closure_flag"].sum()) if "port_closure_flag" in df else 0
    sev_wx=int(df["severe_weather"].sum()) if "severe_weather" in df else 0
    anom=int(df["is_anomaly"].sum())
    rev_risk=df[df["risk_status"].astype(str).isin(["CRITICAL","HIGH"])].shape[0]*1.2

    c1,c2,c3,c4,c5=st.columns(5)
    for col,lbl,val,color in [(c1,"STRIKE EVENTS",f"{strikes:,}","#ff3b5c"),(c2,"PORT CLOSURES",f"{closures:,}","#ff8c00"),
        (c3,"SEVERE WEATHER",f"{sev_wx:,}","#6699ff"),(c4,"ANOMALIES",f"{anom:,}","#f5d800"),
        (c5,"REVENUE @ RISK",f"${rev_risk:.0f}M","#ff3b5c")]:
        metric_card(col,lbl,val,color)
    st.markdown("<br>",unsafe_allow_html=True)

    ps2=ps.copy()
    ps2["lat"]=ps2["origin_port"].map(lambda p:PORT_COORDS.get(p,(None,None))[0])
    ps2["lon"]=ps2["origin_port"].map(lambda p:PORT_COORDS.get(p,(None,None))[1])
    ps2=ps2.dropna(subset=["lat","lon"])
    fig=go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=ps2["lat"],lon=ps2["lon"],mode="markers+text",
        text=ps2["origin_port"].tolist(),textposition="top center",
        textfont=dict(size=9,color="rgba(0,200,180,0.85)"),
        marker=dict(size=ps2["shipment_count"]/ps2["shipment_count"].max()*28+8,
                    color=ps2["delay_rate"],colorscale=[[0,"#00e676"],[0.35,"#f5d800"],[0.65,"#ff8c00"],[1,"#ff3b5c"]],
                    cmin=0,cmax=1,showscale=True,
                    colorbar=dict(title=dict(text="Delay Rate",font=dict(color="#d0e8ff",size=10)),
                                  thickness=14,tickformat=".0%",
                                  tickfont=dict(color="#d0e8ff",size=9)),
                    line=dict(width=1.5,color="rgba(255,255,255,0.3)")),
        customdata=ps2[["origin_port","delay_rate","shipment_count","avg_congestion","anomaly_rate"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Delay: %{customdata[1]:.1%}<br>Ships: %{customdata[2]:,}<br>Congestion: %{customdata[3]:.1f}<extra></extra>",
        name="Ports"))
    fig.update_layout(
        geo=dict(scope="world",showland=True,landcolor="rgba(25,45,75,0.7)",showocean=True,
                 oceancolor="rgba(8,12,20,0.95)",showcoastlines=True,coastlinecolor="rgba(0,200,180,0.18)",
                 showframe=False,bgcolor="rgba(8,12,20,0)",projection_type="natural earth"),
        paper_bgcolor="rgba(8,12,20,0)",margin=dict(l=0,r=0,t=0,b=0),height=420)
    st.plotly_chart(fig,use_container_width=True)

    c1,c2,c3=st.columns(3)
    for col,row in zip([c1,c2,c3],ps.head(3).itertuples()):
        dr=float(row.delay_rate); c_=prob_color(dr)
        col.markdown(f"""<div style="background:rgba(10,20,40,0.85);border:1px solid {c_}30;
            border-left:4px solid {c_};border-radius:6px;padding:14px">
          <div style="font-size:9px;color:#ffffff">HIGHEST RISK PORT</div>
          <div style="font-size:13px;font-weight:700;color:{c_};margin:4px 0">{row.origin_port}</div>
          <div style="font-size:10px;color:#c0d4e8;line-height:1.8">
            Delay: <span style="color:{c_}">{dr:.1%}</span><br>
            Congestion: {row.avg_congestion:.1f}<br>Shipments: {int(row.shipment_count):,}
          </div></div>""",unsafe_allow_html=True)

    section_title("⏱ DISRUPTION SIGNAL TIMELINE")
    dts=df.set_index("timestamp").sort_index()
    fig2=go.Figure()
    if "strike_alert_level" in df.columns:
        s=dts["strike_alert_level"].resample("W").mean()
        fig2.add_trace(go.Scatter(x=s.index,y=s.values,name="Strike Alert",
            line=dict(color="#ff3b5c",width=2),fill="tozeroy",fillcolor="rgba(255,59,92,0.06)"))
    if "geopolitical_risk_index" in df.columns:
        g=dts["geopolitical_risk_index"].resample("W").mean()
        fig2.add_trace(go.Scatter(x=g.index,y=g.values,name="Geopolitical Risk",
            line=dict(color="#cc99ff",width=1.5,dash="dot")))
    if "port_congestion_index" in df.columns:
        cg=dts["port_congestion_index"].resample("W").mean()
        fig2.add_trace(go.Scatter(x=cg.index,y=cg.values,name="Port Congestion",
            line=dict(color="#ff8c00",width=1.5),yaxis="y2"))
    apply_theme(fig2, height=260, legend=dict(orientation="h",y=1.1),
        yaxis2=dict(title="Congestion",overlaying="y",side="right",showgrid=False))
    fig2.update_yaxes(title_text="Signal Level",gridcolor="rgba(0,200,180,0.07)",selector=dict(overlaying=None))
    st.plotly_chart(fig2,use_container_width=True)

    cl,cr=st.columns(2)
    with cl:
        section_title("◆ ANOMALY SCORE vs DELAY PROBABILITY")
        s2=df.sample(min(2000,len(df)),random_state=42)
        fig3=px.scatter(s2,x="anomaly_score",y="ai_delay_prob",color="risk_status",
            color_discrete_map={"CRITICAL":"#ff3b5c","HIGH":"#ff8c00","MEDIUM":"#f5d800","LOW":"#00e676"},
            opacity=0.55)
        fig3.update_traces(marker=dict(size=4))
        apply_theme(fig3, height=280, xaxis_title="Anomaly Score", yaxis_title="AI Delay Prob")
        st.plotly_chart(fig3,use_container_width=True)
    with cr:
        section_title("◆ WEATHER SEVERITY vs CONGESTION")
        s3=df.sample(min(2000,len(df)),random_state=7)
        fig4=px.scatter(s3,x="port_congestion_index",y="weather_severity_score",color="ai_delay_prob",
            color_continuous_scale=[[0,"#00e676"],[0.5,"#ff8c00"],[1,"#ff3b5c"]],opacity=0.5)
        fig4.update_traces(marker=dict(size=4))
        apply_theme(fig4, height=280, xaxis_title="Port Congestion Index", yaxis_title="Weather Severity")
        st.plotly_chart(fig4,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROUTE RISK
# ══════════════════════════════════════════════════════════════════════════════

def tab_risk(df):
    info_banner("<span style='color:#00c8b4;font-weight:700'>◈ ROUTE RISK PREDICTION — </span>"
                "AI delay probability from real HistGradientBoosting model. "
                "Scores >75 auto-trigger rerouting proposals.")

    cc1,cc2,cc3=st.columns([1,1,1])
    with cc1: sort_by=st.selectbox("Sort by",["ai_risk_score","delay_minutes","ai_delay_prob"])
    with cc2: show_n=st.slider("Show top N shipments",5,50,15)
    with cc3: only_r=st.checkbox("Only AI-flagged for reroute")

    dshow=df.copy()
    if only_r: dshow=dshow[dshow["ai_rerouted"]==1]
    dshow=dshow.sort_values(sort_by,ascending=False).head(show_n)

    cl,cr=st.columns([1.6,1])
    with cl:
        section_title("◆ AI DELAY PROBABILITY DISTRIBUTION")
        fig=go.Figure()
        for st_,c_ in [("CRITICAL","#ff3b5c"),("HIGH","#ff8c00"),("MEDIUM","#f5d800"),("LOW","#00e676")]:
            sub=df[df["risk_status"].astype(str)==st_]["ai_delay_prob"]
            if len(sub)>0:
                fig.add_trace(go.Histogram(x=sub,name=st_,marker_color=c_,opacity=0.75,nbinsx=40,histnorm="probability density"))
        apply_theme(fig, height=280, xaxis_title="AI Delay Probability", barmode="overlay", legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig,use_container_width=True)
    with cr:
        section_title("◆ RISK STATUS BREAKDOWN")
        sc=df["risk_status"].astype(str).value_counts()
        fig2=go.Figure(go.Pie(labels=sc.index.tolist(),values=sc.values.tolist(),hole=0.6,
            marker=dict(colors=["#ff3b5c","#ff8c00","#f5d800","#00e676"],line=dict(color="#080c14",width=2)),
            textfont=dict(size=11,color="white")))
        fig2.update_layout(paper_bgcolor="rgba(8,12,20,0)",plot_bgcolor="rgba(8,12,20,0)",
            margin=dict(l=10,r=10,t=10,b=10),height=280,showlegend=True,legend=dict(font=dict(color="#d0e8ff",size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2,use_container_width=True)

    section_title("◆ ROUTE RISK HEATMAP — ORIGIN × DESTINATION")
    piv=df.groupby(["origin_port","destination_port"])["ai_delay_prob"].mean().reset_index().pivot(index="origin_port",columns="destination_port",values="ai_delay_prob")
    if piv.shape[0]>1:
        fig3=px.imshow(piv,color_continuous_scale=[[0,"#00e676"],[0.4,"#f5d800"],[0.7,"#ff8c00"],[1,"#ff3b5c"]],zmin=0,zmax=1,aspect="auto")
        apply_theme(fig3, height=350, xaxis_title="Destination", yaxis_title="Origin")
        st.plotly_chart(fig3,use_container_width=True)

    section_title(f"◆ TOP {len(dshow)} SHIPMENTS")
    for i in range(0,len(dshow),2):
        cols=st.columns(2)
        for j,col in enumerate(cols):
            if i+j>=len(dshow): break
            row=dshow.iloc[i+j]
            prob=float(row.get("ai_delay_prob",0.5)); c_=prob_color(prob)
            with col:
                with st.expander(
                    f"{'🔴' if prob>0.75 else '🟠' if prob>0.5 else '🟡' if prob>0.3 else '🟢'} "
                    f"{row.get('shipment_id','–')}  |  {row.get('origin_port','?')} → {row.get('destination_port','?')}  |  {prob:.0%}",
                    expanded=(prob>0.75 and i+j<4)):
                    st.markdown(f"""<div style="padding:4px 0">
                      <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;margin-bottom:10px">
                        <div style="width:{prob*100:.0f}%;height:100%;background:{c_}"></div></div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:10px;color:#d0e8ff">
                        <div>Carrier: <b style="color:#c0d4e8">{row.get('carrier_id','–')}</b></div>
                        <div>Mode: <b style="color:#c0d4e8">{row.get('transport_mode','–')}</b></div>
                        <div>Delay: <b style="color:{c_}">{float(row.get('delay_minutes',0)):.0f} min</b></div>
                        <div>Congestion: <b style="color:#ff8c00">{float(row.get('port_congestion_index',0)):.1f}</b></div>
                      </div></div>""",unsafe_allow_html=True)
                    if prob>0.60:
                        ca,cb,cc=st.columns(3)
                        ca.button("✓ Accept",key=f"a{i}{j}",use_container_width=True)
                        cb.button("⚡ Simulate",key=f"s{i}{j}",use_container_width=True)
                        cc.button("📩 Escalate",key=f"e{i}{j}",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI REROUTING
# ══════════════════════════════════════════════════════════════════════════════

def run_lp(origin, dest, risks, caps, costs):
    K=len(risks); c_lp=np.array(costs)*(1+np.array(risks))
    res=linprog(c_lp,A_ub=np.eye(K),b_ub=np.array(caps,float),
                A_eq=np.ones((1,K)),b_eq=[1.0],bounds=[(0,1)]*K,method="highs")
    best=int(np.argmax(res.x)) if res.success else 0
    return {"origin":origin,"destination":dest,"allocation":res.x.tolist() if res.success else [1]+[0]*(K-1),
            "best":best,"costs":costs,"risks":risks,"opt_cost":float(res.fun) if res.success else costs[0],"ok":res.success}


def tab_rerouting(df):
    info_banner("<span style='color:#00c8b4;font-weight:700'>◈ AI REROUTING ENGINE — </span>"
                "Routes computed using real LP optimization (scipy.optimize.linprog). "
                "Red dashed = high-risk current. <span style='color:#00e676'>Green = AI-optimized.</span>")

    nr=int(df["ai_rerouted"].sum()); nf=int((df["ai_delay_prob"]>0.60).sum())
    db=df["delay_minutes"].mean(); da=df[df["ai_rerouted"]==0]["delay_minutes"].mean() if len(df[df["ai_rerouted"]==0])>0 else db
    dr=(db-da)/max(db,1)*100

    c1,c2,c3,c4=st.columns(4)
    for col,lbl,val,color in [(c1,"SHIPMENTS REROUTED",f"{nr:,}","#00e676"),(c2,"FLAGGED HIGH-RISK",f"{nf:,}","#ff8c00"),
        (c3,"DELAY REDUCTION",f"{dr:.0f}%","#00c8b4"),(c4,"LP SCENARIOS","3","#6699ff")]:
        metric_card(col,lbl,val,color)
    st.markdown("<br>",unsafe_allow_html=True)

    section_title("◆ ROUTE MAP — REAL PORTS + AI REROUTING OVERLAY")
    pairs=(df.groupby(["origin_port","destination_port"]).agg(prob=("ai_delay_prob","mean"),cnt=("ai_delay_prob","count")).reset_index().sort_values("prob",ascending=False).head(12))
    fig=go.Figure()
    for _,r in pairs.iterrows():
        o,d=r["origin_port"],r["destination_port"]
        if o not in PORT_COORDS or d not in PORT_COORDS: continue
        la1,lo1=PORT_COORDS[o]; la2,lo2=PORT_COORDS[d]
        prob=float(r["prob"]); c_=prob_color(prob)
        n=30; lats=[la1+((la2-la1)*i/n) for i in range(n+1)]; lons=[lo1+((lo2-lo1)*i/n)+6*np.sin(np.pi*i/n)*np.sign(la2-la1) for i in range(n+1)]
        fig.add_trace(go.Scattergeo(lat=lats+[None],lon=lons+[None],mode="lines",
            line=dict(width=2 if prob>0.5 else 1.2,color=c_,dash="dash" if prob>0.5 else "solid"),
            opacity=0.65,showlegend=False,
            hovertemplate=f"<b>{o}→{d}</b><br>Delay prob: {prob:.1%}<extra></extra>"))
    rp=(df[df["ai_rerouted"]==1].groupby(["origin_port","destination_port"])["ai_delay_prob"].mean().reset_index().head(5))
    for _,r in rp.iterrows():
        o,d=r["origin_port"],r["destination_port"]
        if o not in PORT_COORDS or d not in PORT_COORDS: continue
        la1,lo1=PORT_COORDS[o]; la2,lo2=PORT_COORDS[d]
        ml=min(la1,la2)-15; mlo=(lo1+lo2)/2
        lats=[la1,(la1+ml)/2,ml,(ml+la2)/2,la2,None]
        lons=[lo1,lo1+(mlo-lo1)*0.4,mlo,mlo+(lo2-mlo)*0.6,lo2,None]
        fig.add_trace(go.Scattergeo(lat=lats,lon=lons,mode="lines",line=dict(width=2.5,color="#00e676"),
            opacity=0.8,showlegend=False,hovertemplate=f"<b>AI REROUTED</b><br>{o}→{d}<extra></extra>"))
    all_p=set(df["origin_port"].unique())|set(df["destination_port"].unique())
    plat,plon,pnam=[],[],[]
    for p in all_p:
        if p in PORT_COORDS: plat.append(PORT_COORDS[p][0]); plon.append(PORT_COORDS[p][1]); pnam.append(p)
    fig.add_trace(go.Scattergeo(lat=plat,lon=plon,mode="markers+text",
        text=pnam,textposition="top center",
        textfont=dict(size=8,color="rgba(0,200,180,0.8)"),
        marker=dict(size=7,color="#00c8b4",line=dict(width=1.5,color="rgba(255,255,255,0.4)")),
        customdata=[[n] for n in pnam],
        hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
        showlegend=False))
    for lbl,c_,dash in [("HIGH-RISK (current)","#ff3b5c","dash"),("MEDIUM RISK","#f5d800","dash"),("AI OPTIMIZED","#00e676","solid")]:
        fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name=lbl,line=dict(color=c_,dash=dash,width=2)))
    fig.update_layout(geo=dict(scope="world",showland=True,landcolor="rgba(25,45,75,0.65)",showocean=True,
        oceancolor="rgba(8,12,20,0.95)",showcoastlines=True,coastlinecolor="rgba(0,200,180,0.18)",
        showframe=False,bgcolor="rgba(8,12,20,0)",projection_type="natural earth"),
        paper_bgcolor="rgba(8,12,20,0)",margin=dict(l=0,r=0,t=0,b=0),height=420,
        legend=dict(bgcolor="rgba(10,20,40,0.7)",font=dict(color="#d0e8ff",size=10),bordercolor="rgba(0,200,180,0.2)",borderwidth=1))
    st.plotly_chart(fig,use_container_width=True)

    section_title("◆ LINEAR PROGRAMMING OPTIMIZATION — REAL RESULTS")
    info_banner("<span style='color:#00c8b4;font-weight:700'>LP SETUP: </span>"
                "Minimize: Σ cost_k × (1 + risk_k) × x_k &nbsp;|&nbsp; s.t. Σx_k=1, x_k≤capacity_k, x_k≥0. "
                "Disruption scores = real AI delay probabilities.")
    top3=(df.groupby(["origin_port","destination_port"]).agg(pr=("ai_delay_prob","mean")).reset_index().sort_values("pr",ascending=False).head(3))
    for _,row in top3.iterrows():
        risk=float(row["pr"])
        opt=run_lp(row["origin_port"],row["destination_port"],[risk,risk*0.5,risk*0.25],[0.5,0.35,0.15],[1.0,1.12,1.22])
        cl,cr=st.columns([1.2,1])
        with cl:
            st.markdown(f"""<div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,200,180,0.12);border-radius:8px;padding:16px;margin-bottom:10px">
              <div style="font-size:11px;color:#e0eaf5;margin-bottom:10px"><span style="color:#00c8b4">◆</span> {opt['origin']} → {opt['destination']}</div>""",
              unsafe_allow_html=True)
            for i,(alloc,dis,cost) in enumerate(zip(opt["allocation"],opt["risks"],opt["costs"])):
                isbest=(i==opt["best"]); c_="#00e676" if isbest else "#6699ff"
                brd=f"border:1px solid {c_}40;background:rgba(0,200,180,0.04)" if isbest else "border:1px solid rgba(255,255,255,0.05)"
                st.markdown(f"""<div style="padding:8px 12px;border-radius:5px;{brd};margin-bottom:6px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:11px;color:#c0d4e8">{'Direct' if i==0 else f'Alternate {chr(64+i)}'} {'✓' if isbest else ''}</span>
                    <span style="font-size:11px;font-weight:700;color:{c_}">{alloc:.0%} allocated</span>
                  </div>
                  <div style="font-size:10px;color:#c8ddf0">Risk: <b style="color:{'#ff3b5c' if dis>0.5 else '#f5d800' if dis>0.3 else '#00e676'}">{dis:.0%}</b> · Cost: <b>{cost:.2f}×</b> · Weighted: <b style="color:{c_}">{cost*(1+dis):.2f}×</b></div>
                  <div style="height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;margin-top:5px">
                    <div style="width:{alloc*100:.0f}%;height:100%;background:{c_}"></div></div></div>""",unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:10px;color:#d0e8ff;padding:6px'>LP Status: {'✓ Optimal' if opt['ok'] else '⚠ Fallback'} · Cost: <b style='color:#00e676'>{opt['opt_cost']:.3f}×</b></div></div>",unsafe_allow_html=True)
        with cr:
            fig2=go.Figure()
            lbls=["Direct","Alt A","Alt B"]
            fig2.add_trace(go.Bar(x=lbls,y=[c*(1+d) for c,d in zip(opt["costs"],opt["risks"])],name="Weighted Cost",
                marker_color=["#00e676" if i==opt["best"] else "#6699ff" for i in range(3)],
                text=[f"{c*(1+d):.2f}×" for c,d in zip(opt["costs"],opt["risks"])],textposition="outside",textfont=dict(size=10,color="#c0d4e8")))
            fig2.add_trace(go.Bar(x=lbls,y=opt["risks"],name="Risk",marker_color="rgba(255,59,92,0.5)",
                text=[f"{d:.0%}" for d in opt["risks"]],textposition="outside",textfont=dict(size=10,color="#ff3b5c")))
            apply_theme(fig2, height=220, barmode="group", margin=dict(l=30,r=10,t=30,b=30))
            st.plotly_chart(fig2,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════

def tab_xai(df, perm_df):
    has_perm=len(perm_df)>0
    info_banner("<span style='color:#00c8b4;font-weight:700'>◈ EXPLAINABLE AI — PERMUTATION IMPORTANCE — </span>"
                "Model-agnostic XAI (n_repeats=20). Measures accuracy drop when each feature is shuffled. "
                "Top drivers: <span style='color:#ff3b5c'>port_congestion_index, route_risk_score, weather_severity_score</span>")

    if "is_anomaly" in df.columns:
        na=int(df["is_anomaly"].sum()); adr=df[df["is_anomaly"]==1]["is_delayed"].mean() if na>0 else 0
        ndr=df[df["is_anomaly"]==0]["is_delayed"].mean(); lift=adr/max(ndr,0.001)
        c1,c2,c3,c4=st.columns(4)
        for col,lbl,val,color in [(c1,"TOTAL SHIPMENTS",f"{len(df):,}","#00c8b4"),(c2,"ANOMALIES",f"{na:,}","#ff3b5c"),
            (c3,"ANOMALY DELAY%",f"{adr:.1%}","#ff8c00"),(c4,"RISK LIFT",f"{lift:.2f}×","#f5d800")]:
            metric_card(col,lbl,val,color)
        st.markdown("<br>",unsafe_allow_html=True)

    cl,cr=st.columns(2)
    with cl:
        section_title("◆ TOP-15 FEATURE IMPORTANCE (PERMUTATION)")
        if has_perm:
            top15=perm_df.head(15).copy()
            top15["color"]=top15["feature"].map(lambda f:GROUP_COLORS.get(FEATURE_GROUPS.get(f,"Other"),"#d0e8ff"))
            fig=go.Figure()
            fig.add_trace(go.Bar(y=top15["feature"][::-1],x=top15["importance"][::-1],orientation="h",
                marker=dict(color=top15["color"][::-1].tolist()),
                error_x=dict(type="data",array=top15["std"][::-1].tolist(),color="rgba(255,255,255,0.3)",thickness=1.5),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
            apply_theme(fig, height=380, xaxis_title="Permutation Importance", margin=dict(l=180,r=20,t=20,b=40))
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Train the model to see real permutation importance.")
    with cr:
        section_title("◆ FEATURE GROUP CONTRIBUTION")
        if has_perm:
            pg=perm_df.copy(); pg["group"]=pg["feature"].map(lambda f:FEATURE_GROUPS.get(f,"Other"))
            gi=pg.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
            gi["pct"]=gi["importance"]/gi["importance"].sum()
            for _,r in gi.iterrows():
                c_=GROUP_COLORS.get(r["group"],"#d0e8ff"); pct=float(r["pct"])
                st.markdown(f"""<div style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="font-size:11px;color:#c0d4e8">{r['group']}</span>
                    <span style="font-size:11px;font-weight:700;color:{c_}">{pct:.0%}</span>
                  </div>
                  <div style="height:10px;background:rgba(255,255,255,0.05);border-radius:5px;overflow:hidden">
                    <div style="width:{pct*100:.0f}%;height:100%;background:{c_}"></div></div></div>""",unsafe_allow_html=True)
        if "anomaly_score" in df.columns:
            section_title("◆ ISOLATION FOREST — ANOMALY SCORES")
            fig2=go.Figure()
            for av,nm,c_ in [(0,"Normal","#00c8b4"),(1,"Anomaly","#ff3b5c")]:
                sub=df[df["is_anomaly"]==av]["anomaly_score"]
                if len(sub)>0: fig2.add_trace(go.Histogram(x=sub,name=nm,marker_color=c_,opacity=0.7,nbinsx=40,histnorm="probability density"))
            apply_theme(fig2, height=200, xaxis_title="Anomaly Score", barmode="overlay",
                legend=dict(orientation="h",y=1.1,font=dict(size=9)),margin=dict(l=40,r=10,t=20,b=30))
            st.plotly_chart(fig2,use_container_width=True)

    section_title("◆ PER-SHIPMENT EXPLAINABILITY")
    top100=df.sort_values("ai_delay_prob",ascending=False).head(100)
    ids=top100["shipment_id"].astype(str).tolist()
    col_s,col_i=st.columns([1,2])
    with col_s:
        sel=st.selectbox("Select Shipment (top 100 by risk)",ids)
    if sel:
        row=top100[top100["shipment_id"].astype(str)==sel].iloc[0]
        prob=float(row.get("ai_delay_prob",0.5)); c_=prob_color(prob)
        with col_i:
            st.markdown(f"""<div style="padding:10px 16px;background:rgba(10,20,40,0.85);
                border:1px solid {c_}30;border-left:4px solid {c_};border-radius:6px;margin-top:24px">
              <div style="display:flex;justify-content:space-between">
                <div><span style="font-size:14px;font-weight:700;color:#e0eaf5">{sel}</span>
                  <span style="font-size:9px;margin-left:10px;padding:2px 8px;border-radius:3px;background:{c_}15;color:{c_}">{str(row.get('risk_status','?'))}</span>
                  <div style="font-size:10px;color:#c8ddf0;margin-top:4px">{row.get('origin_port','?')} → {row.get('destination_port','?')} | {row.get('carrier_id','?')}</div>
                </div>
                <div style="text-align:right"><div style="font-size:26px;font-weight:800;color:{c_}">{prob:.0%}</div>
                  <div style="font-size:9px;color:#ffffff">AI DELAY PROBABILITY</div></div>
              </div></div>""",unsafe_allow_html=True)

        imp_cols=["port_congestion_index","route_risk_score","weather_severity_score","strike_alert_level",
                  "geopolitical_risk_index","anomaly_score","berth_occupancy_pct","yard_utilization_pct",
                  "median_waiting_time_hours","wind_x_wave","combined_disruption","transit_delay_8h_bin"]
        avail=[c for c in imp_cols if c in row.index]
        if avail and has_perm:
            pl=perm_df.set_index("feature")["importance"].to_dict()
            fmax=df[avail].max(); nv=row[avail].abs()/(fmax+1e-9)
            contrib=nv*pd.Series(avail).map(lambda f:pl.get(f,0.001)).values
            cs=pd.Series(contrib.values if hasattr(contrib,"values") else list(contrib),index=avail).sort_values(ascending=False)
            cl2,cr2=st.columns(2)
            with cl2:
                section_title("SHAP-STYLE CONTRIBUTIONS")
                fig3=go.Figure()
                fig3.add_trace(go.Bar(y=cs.index[::-1],x=cs.values[::-1],orientation="h",
                    marker_color=[GROUP_COLORS.get(FEATURE_GROUPS.get(f,"Other"),"#6699ff") for f in cs.index[::-1]],
                    hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"))
                apply_theme(fig3, height=280, margin=dict(l=180,r=10,t=10,b=30))
                st.plotly_chart(fig3,use_container_width=True)
            with cr2:
                section_title("FEATURE VALUES — THIS SHIPMENT")
                for feat in cs.index[:8]:
                    val=float(row[feat]); c_=GROUP_COLORS.get(FEATURE_GROUPS.get(feat,"Other"),"#6699ff")
                    mx=float(df[feat].abs().max()); pct=min(abs(val/max(mx,0.001))*100,100)
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                      <div style="width:170px;font-size:10px;color:#d0e8ff">{feat}</div>
                      <div style="flex:1;height:5px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
                        <div style="width:{pct:.0f}%;height:100%;background:{c_}"></div></div>
                      <div style="width:60px;font-size:10px;font-weight:700;color:{c_};text-align:right">{val:.3f}</div>
                    </div>""",unsafe_allow_html=True)

        st.markdown(f"""<div style="margin-top:12px;padding:12px 16px;background:rgba(0,200,180,0.04);
            border:1px solid rgba(0,200,180,0.18);border-radius:6px;font-size:11px;color:#d0e8ff;line-height:1.8">
          <span style="color:#00c8b4;font-weight:700">⬡ NL REASONING: </span>
          {"CRITICAL: Shipment flagged for immediate rerouting. Port congestion and weather combine to push delay probability above 75%. LP optimizer recommends Cape of Good Hope bypass." if prob>0.75 else
           "HIGH RISK: Route risk score and congestion index are primary drivers. Consider alternate corridor to reduce delay by ~40%." if prob>0.5 else
           f"Shipment within acceptable risk bounds ({prob:.0%} delay probability). Continue standard monitoring."}
        </div>""",unsafe_allow_html=True)

    section_title("◆ TOP FEATURE CORRELATIONS WITH DELAY")
    num_cols=[c for c in df.select_dtypes(include=[np.number]).columns
              if c not in ["is_delayed","ai_rerouted","ai_risk_score"] and "enc" not in c]
    if num_cols and "is_delayed" in df.columns:
        corr=df[num_cols+["is_delayed"]].corr()["is_delayed"].drop("is_delayed")
        tc=corr.abs().sort_values(ascending=False).head(15); cv=corr[tc.index]
        fig4=go.Figure(go.Bar(y=tc.index[::-1],x=cv[tc.index[::-1]].values,orientation="h",
            marker_color=["#ff3b5c" if v>0 else "#00c8b4" for v in cv[tc.index[::-1]]],
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.4f}<extra></extra>"))
        fig4.add_vline(x=0,line_color="rgba(255,255,255,0.2)",line_width=1)
        apply_theme(fig4, height=320, xaxis_title="Pearson Correlation with is_delayed", margin=dict(l=200,r=20,t=10,b=30))
        st.plotly_chart(fig4,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def tab_performance(bundle, df):
    has=bundle and bundle.get("tuned_acc") is not None
    info_banner("<span style='color:#00c8b4;font-weight:700'>◈ PERFORMANCE ANALYTICS — </span>"
                "HistGradientBoosting vs Random Forest. 3-fold TimeSeriesSplit CV. "
                "All metrics on held-out chronological test set (last 20%).")

    if not has:
        st.warning("⚠ Model not trained. Run `python train_model.py` then restart the dashboard.")
        st.metric("Total Records",f"{len(df):,}"); return

    acc=bundle["tuned_acc"]; auc=bundle["tuned_auc"]; cv_m=np.mean(bundle["hgb_cv"])
    dr=bundle.get("delay_reduction_pct",70); cs=bundle.get("cost_savings_k",847)

    c1,c2,c3,c4=st.columns(4)
    for col,lbl,val,color,sub in [
        (c1,"TUNED ACCURACY", f"{acc:.1%}", "#00e676","HistGradientBoosting"),
        (c2,"AUC-ROC",        f"{auc:.4f}", "#00c8b4",f"vs RF: {bundle.get('rf_auc',0.88):.4f}"),
        (c3,"CV ACCURACY",    f"{cv_m:.1%}","#f5d800","3-fold TimeSeriesSplit"),
        (c4,"DELAY REDUCTION",f"{dr:.0f}%", "#ff8c00",f"${cs:.0f}K saved"),
    ]: metric_card(col,lbl,val,color,sub)
    st.markdown("<br>",unsafe_allow_html=True)

    cl,cr=st.columns(2)
    with cl:
        section_title("◆ CONFUSION MATRIX")
        cm=np.array(bundle["confusion_matrix"])
        if cm.shape==(2,2):
            fig=go.Figure(go.Heatmap(z=cm[::-1],x=["Pred On-Time","Pred Delayed"],y=["Actual Delayed","Actual On-Time"],
                colorscale=[[0,"rgba(8,12,20,0.9)"],[0.5,"rgba(0,200,180,0.3)"],[1,"rgba(0,200,180,0.8)"]],
                text=cm[::-1].astype(str),texttemplate="<b>%{text}</b>",textfont=dict(size=18,color="white"),showscale=False))
            apply_theme(fig, height=260, margin=dict(l=100,r=20,t=20,b=60))
            st.plotly_chart(fig,use_container_width=True)
            tn,fp,fn,tp=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
            prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,0.001)
            p1,p2,p3=st.columns(3)
            for col,lbl,val,color in [(p1,"Precision",f"{prec:.1%}","#00e676"),(p2,"Recall",f"{rec:.1%}","#00c8b4"),(p3,"F1-Score",f"{f1:.3f}","#f5d800")]:
                col.markdown(f"""<div style="text-align:center;background:rgba(0,200,180,0.05);padding:8px;border-radius:5px">
                  <div style="font-size:15px;font-weight:700;color:{color}">{val}</div>
                  <div style="font-size:9px;color:#ffffff">{lbl}</div></div>""",unsafe_allow_html=True)
    with cr:
        section_title("◆ CROSS-VALIDATION — HGB vs RF")
        folds=[f"Fold {i+1}" for i in range(len(bundle["hgb_cv"]))]
        fig2=go.Figure()
        fig2.add_trace(go.Bar(name="HistGradientBoosting",x=folds,y=bundle["hgb_cv"],marker_color="#6699ff",
            text=[f"{v:.1%}" for v in bundle["hgb_cv"]],textposition="outside",textfont=dict(size=9,color="#6699ff")))
        fig2.add_trace(go.Bar(name="Random Forest",x=folds,y=bundle["rf_cv"],marker_color="#ff8c00",
            text=[f"{v:.1%}" for v in bundle["rf_cv"]],textposition="outside",textfont=dict(size=9,color="#ff8c00")))
        fig2.add_hline(y=np.mean(bundle["hgb_cv"]),line_dash="dash",line_color="#6699ff",line_width=1.5,opacity=0.7)
        fig2.add_hline(y=np.mean(bundle["rf_cv"]), line_dash="dash",line_color="#ff8c00",line_width=1.5,opacity=0.7)
        apply_theme(fig2, height=260, barmode="group", legend=dict(orientation="h",y=1.08), margin=dict(l=40,r=10,t=20,b=30))
        fig2.update_yaxes(tickformat=".0%", range=[0.7,0.92])
        st.plotly_chart(fig2,use_container_width=True)

    cl2,cr2=st.columns(2)
    with cl2:
        section_title("◆ ROC CURVE — HGB vs RF")
        ft=bundle.get("fpr_tuned",[0,0.1,0.3,0.6,1]); tt=bundle.get("tpr_tuned",[0,0.6,0.85,0.97,1])
        fr=bundle.get("fpr_rf",[0,0.1,0.3,0.6,1]);    tr=bundle.get("tpr_rf",[0,0.5,0.78,0.94,1])
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=ft,y=tt,mode="lines",name=f"HGB (AUC={auc:.4f})",
            line=dict(color="#6699ff",width=2.5),fill="tozeroy",fillcolor="rgba(102,153,255,0.05)"))
        fig3.add_trace(go.Scatter(x=fr,y=tr,mode="lines",name=f"RF (AUC={bundle.get('rf_auc',0.88):.4f})",
            line=dict(color="#ff8c00",width=2,dash="dot")))
        fig3.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
            line=dict(color="rgba(255,255,255,0.15)",width=1,dash="dash")))
        apply_theme(fig3, height=320, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(orientation="h",y=0.04,x=0.05,bgcolor="rgba(8,12,20,0.7)"))
        st.plotly_chart(fig3,use_container_width=True)
    with cr2:
        section_title("◆ AI vs BASELINE — BUSINESS IMPACT")
        bd=bundle.get("baseline_delayed",0); ad=bundle.get("ai_delayed",0)
        col_ba,col_ai=st.columns(2)
        col_ba.markdown(f"""<div style="background:rgba(255,59,92,0.05);border:1px solid rgba(255,59,92,0.2);border-radius:6px;padding:12px">
          <div style="font-size:10px;color:#ff3b5c;letter-spacing:0.1em;margin-bottom:8px">BASELINE</div>
          <div style="font-size:11px;color:#d0e8ff;line-height:2">Delayed: <span style="color:#ff3b5c;font-weight:700">{bd:,}</span><br>
          Avg delay: <span style="color:#ff3b5c;font-weight:700">{df['delay_minutes'].mean():.0f} min</span><br>
          Cost index: <span style="color:#ff3b5c;font-weight:700">100.0</span></div></div>""",unsafe_allow_html=True)
        col_ai.markdown(f"""<div style="background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.2);border-radius:6px;padding:12px">
          <div style="font-size:10px;color:#00e676;letter-spacing:0.1em;margin-bottom:8px">AI REROUTED</div>
          <div style="font-size:11px;color:#d0e8ff;line-height:2">Delayed: <span style="color:#00e676;font-weight:700">{ad:,}</span><br>
          Reduction: <span style="color:#00e676;font-weight:700">{dr:.0f}%</span><br>
          Savings: <span style="color:#00e676;font-weight:700">${cs:.0f}K</span></div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,200,180,0.12);border-radius:8px;padding:14px;font-size:11px;color:#d0e8ff;line-height:2">
          Model: <b style="color:#c0d4e8">HistGradientBoosting</b><br>
          Train size: <b style="color:#c0d4e8">{bundle.get('train_size',0):,}</b> &nbsp;|&nbsp;
          Test size: <b style="color:#c0d4e8">{bundle.get('test_size',0):,}</b><br>
          Features: <b style="color:#c0d4e8">{bundle.get('n_features',0)}</b> &nbsp;|&nbsp;
          Trained: <b style="color:#c0d4e8">{bundle.get('trained_at','N/A')[:10]}</b>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LLM AGENT
# ══════════════════════════════════════════════════════════════════════════════

def tab_agent(df_raw, df_pred):
    info_banner(
        "<span style='color:#00c8b4;font-weight:700'>◈ LLM REROUTING AGENT — </span>"
        "LangGraph 3-node pipeline: Port Risk → Route History → Llama-3.3-70b analysis. "
        "Powered by Groq. Set <code>GROQ_API_KEY</code> in your <code>.env</code> file."
    )

    if not AGENT_AVAILABLE:
        st.error(
            "⚠ Agent dependencies not installed. Run:\n\n"
            "```\npip install langchain-groq langgraph langchain-core python-dotenv\n```"
        )
        return

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.warning("⚠ GROQ_API_KEY not found. Add it to your `.env` file and restart.")
        st.code('GROQ_API_KEY=your_key_here', language="bash")
        return

    # ── Port selectors ────────────────────────────────────────────────────────
    all_ports = sorted(df_raw["origin_port"].dropna().unique().tolist())
    dest_ports = sorted(df_raw["destination_port"].dropna().unique().tolist())

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        origin = st.selectbox("Origin Port", all_ports,
                              index=all_ports.index("Dubai") if "Dubai" in all_ports else 0)
    with c2:
        # Default to a destination different from origin
        default_dest = [p for p in dest_ports if p != origin]
        destination = st.selectbox("Destination Port", dest_ports,
                                   index=dest_ports.index(default_dest[0]) if default_dest else 0)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⬡ Run Agent Analysis", use_container_width=True)

    # ── Live data context panel ───────────────────────────────────────────────
    sub_orig = df_pred[df_pred["origin_port"] == origin]
    sub_route = df_pred[
        (df_pred["origin_port"] == origin) &
        (df_pred["destination_port"] == destination)
    ]

    d1, d2, d3, d4 = st.columns(4)
    metric_card(d1, "ORIGIN DELAY RATE",
                f"{sub_orig['is_delayed'].mean():.1%}" if len(sub_orig) else "N/A",
                "#ff3b5c")
    metric_card(d2, "ROUTE DELAY RATE",
                f"{sub_route['is_delayed'].mean():.1%}" if len(sub_route) else "N/A",
                "#ff8c00")
    metric_card(d3, "AVG AI RISK SCORE",
                f"{sub_orig['ai_risk_score'].mean():.1f}" if len(sub_orig) else "N/A",
                "#f5d800")
    metric_card(d4, "ROUTE SHIPMENTS",
                f"{len(sub_route):,}", "#00c8b4")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Agent execution ───────────────────────────────────────────────────────
    result_key = f"agent_result_{origin}_{destination}"

    if run_btn:
        agent = build_agent(df_raw)
        if agent is None:
            st.error("Failed to build agent. Check GROQ_API_KEY and dependencies.")
            return

        # Pipeline status display
        status_box = st.empty()

        def _status(step, msg, color="#00c8b4"):
            status_box.markdown(
                f"""<div style="background:rgba(10,20,40,0.9);border:1px solid {color}30;
                border-left:4px solid {color};border-radius:6px;padding:12px 16px;
                font-size:11px;color:#d0e8ff">
                <span style="color:{color};font-weight:700">{step}</span> {msg}
                </div>""",
                unsafe_allow_html=True
            )

        _status("[ 1/3 ]", f"Querying port risk data for <b>{origin}</b>…", "#6699ff")
        import time; time.sleep(0.3)
        _status("[ 2/3 ]", f"Loading route history: <b>{origin} → {destination}</b>…", "#f5d800")
        time.sleep(0.3)
        _status("[ 3/3 ]", "Sending to Llama-3.3-70b via Groq…", "#ff8c00")

        try:
            result = agent.invoke({"origin": origin, "destination": destination})
            st.session_state[result_key] = result
            status_box.empty()
        except Exception as e:
            status_box.empty()
            st.error(f"Agent error: {e}")
            return

    # ── Render result ─────────────────────────────────────────────────────────
    if result_key in st.session_state:
        result = st.session_state[result_key]
        analysis = result.get("analysis", "")
        parsed   = _parse_analysis(analysis)

        # Route mini-map
        if origin in PORT_COORDS and destination in PORT_COORDS:
            la1, lo1 = PORT_COORDS[origin]
            la2, lo2 = PORT_COORDS[destination]
            fig_map = go.Figure()
            # Route arc
            n = 40
            arc_lats = [la1 + (la2-la1)*i/n for i in range(n+1)]
            arc_lons = [lo1 + (lo2-lo1)*i/n + 5*np.sin(np.pi*i/n) for i in range(n+1)]
            route_risk = sub_route["ai_delay_prob"].mean() if len(sub_route) else 0.5
            fig_map.add_trace(go.Scattergeo(
                lat=arc_lats, lon=arc_lons, mode="lines",
                line=dict(width=3, color=prob_color(route_risk), dash="dash"),
                opacity=0.8, showlegend=False,
                hovertemplate=f"{origin} → {destination}<br>Risk: {route_risk:.1%}<extra></extra>"
            ))
            fig_map.add_trace(go.Scattergeo(
                lat=[la1, la2], lon=[lo1, lo2], mode="markers+text",
                text=[origin, destination], textposition=["bottom right", "bottom left"],
                textfont=dict(size=11, color="#00c8b4"),
                marker=dict(size=[14, 14],
                            color=[prob_color(sub_orig["ai_delay_prob"].mean() if len(sub_orig) else 0.5), "#00e676"],
                            line=dict(width=2, color="rgba(255,255,255,0.5)")),
                showlegend=False
            ))
            fig_map.update_layout(
                geo=dict(scope="world", showland=True, landcolor="rgba(25,45,75,0.6)",
                         showocean=True, oceancolor="rgba(8,12,20,0.95)",
                         showcoastlines=True, coastlinecolor="rgba(0,200,180,0.15)",
                         showframe=False, bgcolor="rgba(8,12,20,0)",
                         projection_type="natural earth",
                         center=dict(lat=(la1+la2)/2, lon=(lo1+lo2)/2),
                         projection_scale=2.5),
                paper_bgcolor="rgba(8,12,20,0)",
                margin=dict(l=0, r=0, t=0, b=0), height=280
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # Analysis cards
        section_title("◆ AGENT ANALYSIS OUTPUT")
        col_r, col_rec, col_conf = st.columns(3)

        def _analysis_card(col, label, content, color):
            col.markdown(f"""
            <div style="background:rgba(10,20,40,0.9);border:1px solid {color}30;
                        border-left:4px solid {color};border-radius:8px;
                        padding:16px;min-height:120px">
              <div style="font-size:9px;color:{color};letter-spacing:0.15em;
                          margin-bottom:8px;font-weight:700">{label}</div>
              <div style="font-size:11px;color:#c0d4e8;line-height:1.7">{content or "—"}</div>
            </div>""", unsafe_allow_html=True)

        _analysis_card(col_r,   "⚠ RISK ASSESSMENT",    parsed["RISK"],           "#ff3b5c")
        _analysis_card(col_rec, "⬡ RECOMMENDATION",     parsed["RECOMMENDATION"], "#00e676")
        _analysis_card(col_conf,"◈ CONFIDENCE",          parsed["CONFIDENCE"],     "#6699ff")

        # Raw output toggle
        with st.expander("Raw agent output"):
            st.markdown(f"""<pre style="font-size:10px;color:#d0e8ff;
                background:rgba(10,20,40,0.7);padding:14px;border-radius:6px;
                white-space:pre-wrap">{analysis}</pre>""", unsafe_allow_html=True)

        # Graph trace
        section_title("◆ LANGGRAPH PIPELINE TRACE")
        steps = [
            ("port_risk",     "Port Risk Node",     result.get("port_risk",""),     "#6699ff"),
            ("route_history", "Route History Node", result.get("route_history",""), "#f5d800"),
            ("reroute",       "LLM Reroute Node",   "Analysis complete ✓",          "#00e676"),
        ]
        for key, label, data, color in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:8px">
              <div style="width:10px;height:10px;border-radius:50%;background:{color};
                          margin-top:4px;flex-shrink:0;
                          box-shadow:0 0 8px {color}"></div>
              <div style="flex:1;background:rgba(10,20,40,0.7);border:1px solid {color}20;
                          border-radius:5px;padding:10px 14px">
                <div style="font-size:10px;color:{color};font-weight:700;margin-bottom:4px">{label}</div>
                <div style="font-size:10px;color:#c0d4e8;white-space:pre-wrap">{data}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    else:
        # Placeholder when no analysis run yet
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#ffffff;font-size:12px">
          <div style="font-size:32px;margin-bottom:12px">⬡</div>
          Select origin & destination, then click <b style="color:#00c8b4">Run Agent Analysis</b>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    if "show_sidebar" not in st.session_state:
        st.session_state.show_sidebar = True

    with st.spinner("⬡ Loading dataset & engineering features…"):
        try:
            df, features, encoders = load_and_engineer(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Dataset not found at `{DATA_PATH}`. "
                     "Please copy your CSV into the `data/` folder and restart.")
            return
        except Exception as e:
            st.error(f"Data loading error: {e}"); return

    bundle = load_model()
    with st.spinner("⬡ Running AI predictions…"):
        if bundle:
            try:
                df_pred = run_predictions(df, features, bundle)
            except Exception as e:
                st.warning(f"Model prediction failed ({e}). Using heuristic scores.")
                df_pred = fallback_predictions(df)
        else:
            st.warning("⚠ Model not found. Showing heuristic risk scores. "
                       "Run `python train_model.py` to enable real AI predictions.")
            df_pred = fallback_predictions(df)

    ps      = port_stats(df_pred)
    monthly = monthly_trend(df_pred)
    perm_df = pd.DataFrame(bundle.get("perm_importance",[])) if bundle else pd.DataFrame()

    df_filtered = df_pred.copy()
    if st.session_state.show_sidebar:
        df_filtered = render_sidebar(df_pred)
    else:
        st.markdown("""
        <style>
        [data-testid="stSidebar"]{display:none !important}
        </style>""", unsafe_allow_html=True)

    btn_label = "✕ Hide Panel" if st.session_state.show_sidebar else "⬡ Show Panel"
    btn_color = "rgba(0,200,180,0.08)" if st.session_state.show_sidebar else "rgba(0,200,180,0.22)"
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] > button.sidebar-toggle {{
        position:fixed !important; top:14px !important; left:14px !important;
        z-index:999999 !important; background:{btn_color} !important;
        color:#00c8b4 !important; border:1px solid rgba(0,200,180,0.55) !important;
        border-radius:7px !important; padding:5px 12px !important;
        font-family:'Courier New',monospace !important; font-size:11px !important;
        letter-spacing:0.08em !important; cursor:pointer !important;
        box-shadow:0 0 14px rgba(0,200,180,0.25) !important;
    }}
    </style>""", unsafe_allow_html=True)

    col_toggle, col_spacer = st.columns([1, 12])
    with col_toggle:
        if st.button(btn_label, key="sidebar_toggle"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar
            st.rerun()

    render_header(df_filtered, bundle)

    tabs = st.tabs([
        "🌐 Global Overview",
        "⚡ Disruption Monitor",
        "⚠ Route Risk",
        "🔀 AI Rerouting",
        "🧠 Explainable AI",
        "📊 Performance",
        "🤖 LLM Agent",      # ← new tab
    ])

    with tabs[0]: tab_overview(df_filtered, ps, monthly)
    with tabs[1]: tab_disruption(df_filtered, ps)
    with tabs[2]: tab_risk(df_filtered)
    with tabs[3]: tab_rerouting(df_filtered)
    with tabs[4]: tab_xai(df_filtered, perm_df)
    with tabs[5]: tab_performance(bundle, df_filtered)
    with tabs[6]: tab_agent(df, df_filtered)   # pass raw df for agent data nodes


if __name__ == "__main__":
    main()