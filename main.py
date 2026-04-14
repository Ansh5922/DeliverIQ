import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliverIQ · Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:         #0a0c10;
    --surface:    #111318;
    --border:     #1e2330;
    --accent:     #f97316;
    --accent2:    #fbbf24;
    --text:       #e8eaf0;
    --muted:      #5a6070;
    --success:    #34d399;
    --card-bg:    #13161d;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

/* Hero */
.hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
}
.hero-badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #fff 0%, var(--accent2) 60%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1.1 !important;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Cards */
.param-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.param-card:hover { border-color: #2d3450; }

/* Result panel */
.result-panel {
    background: linear-gradient(135deg, #13161d 0%, #1a1e2a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(249,115,22,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-unit {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* Category badge */
.cat-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}
.cat-fast  { background: rgba(52,211,153,0.15); color: var(--success); border: 1px solid rgba(52,211,153,0.3); }
.cat-avg   { background: rgba(251,191,36,0.15);  color: var(--accent2); border: 1px solid rgba(251,191,36,0.3); }
.cat-slow  { background: rgba(249,115,22,0.15);  color: var(--accent);  border: 1px solid rgba(249,115,22,0.3); }

/* Stat chips */
.chip-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1.5rem; }
.chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-size: 0.72rem;
    color: var(--muted);
}
.chip span { color: var(--text); font-weight: 500; }

/* Streamlit widget overrides */
.stSlider > div > div > div { background: var(--accent) !important; }
div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Confidence bar */
.conf-bar-wrap {
    background: var(--surface);
    border-radius: 99px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("best_random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_artifacts()

# Encoding maps (alphabetical label-encoding as used during training)
WEATHER_MAP     = {"Clear": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Windy": 4}
TRAFFIC_MAP     = {"High": 0, "Low": 1, "Medium": 2}
TIME_MAP        = {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3}
VEHICLE_MAP     = {cls: idx for idx, cls in enumerate(le.classes_)}  # Bike=0, Car=1, Scooter=2

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-badge">🛵 &nbsp;Powered by Random Forest</div>
    <h1>DeliverIQ</h1>
    <div class="hero-sub">Intelligent delivery time prediction · Configure your order parameters below</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Route Parameters ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">01 &nbsp; Route Parameters</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        distance = st.slider("Distance (km)", min_value=0.5, max_value=20.0,
                             value=10.0, step=0.1, format="%.1f km")
    with c2:
        prep_time = st.slider("Preparation Time (min)", min_value=5, max_value=29,
                              value=15, step=1, format="%d min")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Conditions ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">02 &nbsp; Conditions</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    WEATHER_ICONS = {"Clear": "☀️", "Foggy": "🌫️", "Rainy": "🌧️", "Snowy": "❄️", "Windy": "💨"}
    TRAFFIC_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    TIME_ICONS    = {"Morning": "🌅", "Afternoon": "🌤️", "Evening": "🌆", "Night": "🌙"}

    with c3:
        weather_options = list(WEATHER_MAP.keys())
        weather_fmt     = [f"{WEATHER_ICONS[w]}  {w}" for w in weather_options]
        weather_sel     = st.selectbox("Weather Condition", weather_fmt,
                                       index=0, key="weather")
        weather         = weather_sel.split("  ")[1]

        time_options = list(TIME_MAP.keys())
        time_fmt     = [f"{TIME_ICONS[t]}  {t}" for t in time_options]
        time_sel     = st.selectbox("Time of Day", time_fmt, index=0, key="time")
        time_of_day  = time_sel.split("  ")[1]

    with c4:
        traffic_options = list(TRAFFIC_MAP.keys())
        traffic_fmt     = [f"{TRAFFIC_ICONS[t]}  {t}" for t in traffic_options]
        traffic_sel     = st.selectbox("Traffic Level", traffic_fmt,
                                       index=0, key="traffic")
        traffic         = traffic_sel.split("  ")[1]

        VEHICLE_ICONS   = {"Bike": "🚲", "Scooter": "🛵", "Car": "🚗"}
        vehicle_options = list(VEHICLE_MAP.keys())
        vehicle_fmt     = [f"{VEHICLE_ICONS[v]}  {v}" for v in vehicle_options]
        vehicle_sel     = st.selectbox("Vehicle Type", vehicle_fmt,
                                       index=2, key="vehicle")
        vehicle         = vehicle_sel.split("  ")[1]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Courier Profile ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">03 &nbsp; Courier Profile</div>', unsafe_allow_html=True)

    experience = st.slider("Courier Experience (years)", min_value=0.0, max_value=9.0,
                           value=4.0, step=0.5, format="%.1f yrs")

    exp_label = "Novice" if experience < 2 else ("Experienced" if experience < 6 else "Expert")
    exp_color = "#f97316" if experience < 2 else ("#fbbf24" if experience < 6 else "#34d399")
    st.markdown(f"<small style='color:{exp_color}; font-size:0.75rem'>◉ {exp_label} courier</small>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Delivery Time")


# ── Right Panel ───────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build feature vector
        feat = np.array([[
            distance,
            prep_time,
            experience,
            WEATHER_MAP[weather],
            TRAFFIC_MAP[traffic],
            TIME_MAP[time_of_day],
            VEHICLE_MAP[vehicle],
        ]])

        prediction = model.predict(feat)[0]
        mins = round(prediction)

        # Category
        if mins <= 40:
            cat, cat_class, cat_icon = "Fast Delivery", "cat-fast", "🚀"
        elif mins <= 70:
            cat, cat_class, cat_icon = "Average Delivery", "cat-avg", "⏱️"
        else:
            cat, cat_class, cat_icon = "Slow Delivery", "cat-slow", "🐢"

        # Confidence proxy: std of individual tree predictions
        tree_preds  = np.array([t.predict(feat)[0] for t in model.estimators_])
        std_dev     = tree_preds.std()
        conf        = max(0, min(100, 100 - (std_dev / prediction) * 100))

        # Speed estimate
        speed_kmh = distance / (prediction / 60) if prediction > 0 else 0

        st.markdown(f"""
        <div class="result-panel">
            <div class="result-label">Estimated Delivery Time</div>
            <div class="result-value">{mins}</div>
            <div class="result-unit">minutes</div>
            <div class="cat-badge {cat_class}">{cat_icon} &nbsp;{cat}</div>

            <div class="chip-row" style="justify-content:center;">
                <div class="chip">🌡️ Confidence <span>{conf:.0f}%</span></div>
                <div class="chip">⚡ Speed <span>{speed_kmh:.1f} km/h</span></div>
                <div class="chip">📦 Prep <span>{prep_time} min</span></div>
                <div class="chip">🛣️ Route <span>{distance:.1f} km</span></div>
            </div>

            <div style="margin-top:1.5rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--muted);margin-bottom:0.3rem;">
                    <span>Model Confidence</span><span>{conf:.0f}%</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{conf:.0f}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Breakdown ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Feature Influence</div>', unsafe_allow_html=True)

        feat_names  = ["Distance", "Prep Time", "Experience", "Weather", "Traffic", "Time of Day", "Vehicle"]
        importances = model.feature_importances_
        max_imp     = importances.max()

        for name, imp in zip(feat_names, importances):
            pct = imp / max_imp * 100
            bar_color = "#f97316" if pct > 60 else ("#fbbf24" if pct > 30 else "#5a6070")
            st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--muted);margin-bottom:0.2rem;">
                    <span>{name}</span><span style="color:var(--text)">{imp*100:.1f}%</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{pct:.0f}%;background:{bar_color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-panel" style="min-height:340px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.75rem;">
            <div style="font-size:3rem;opacity:0.4">🛵</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:var(--muted);text-align:center;">
                Configure parameters<br>and hit <span style="color:var(--accent)">Predict</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:var(--muted);font-size:0.7rem;letter-spacing:0.08em;border-top:1px solid var(--border);padding-top:1rem;">
    DeliverIQ &nbsp;·&nbsp; Random Forest Regressor &nbsp;·&nbsp; 50 Estimators · Max Depth 10
</div>
""", unsafe_allow_html=True)