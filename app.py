import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import time

# Session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "splash"

# =========================
# SPLASH SCREEN
# =========================
if st.session_state.page == "splash":

    st.markdown("""
        <style>
        .main {
            text-align: center;
            padding-top: 100px;
        }
        .title {
            font-size: 50px;
            color: #00FFAA;
            font-weight: bold;
        }
        .subtitle {
            font-size: 20px;
            color: white;
            margin-top: 20px;
        }
        .box {
            margin-top: 40px;
            padding: 20px;
            border-radius: 15px;
            background-color: #161A23;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">🚖 Taxi Demand Prediction System</div>', unsafe_allow_html=True)

    st.markdown('<div class="subtitle">Predicting taxi demand using Machine Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box">
    <h3>📄 Overview</h3>
    <p>
    This application predicts taxi demand using historical trip data and machine learning techniques.
    It helps identify peak hours and optimize transportation services efficiently.
    </p>

    <h3>⚙️ Features</h3>
    <p>
    ✔ Real-time Predictions<br>
    ✔ XGBoost Model<br>
    ✔ Interactive Dashboard<br>
    ✔ Dark Theme UI
    </p>

   
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Enter Dashboard"):
        st.session_state.page = "main"

    st.markdown('</div>', unsafe_allow_html=True)
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Taxi Demand Predictor",
    page_icon="🚖",
    layout="wide"
)

# =========================
# ULTIMATE DARK THEME CSS
# =========================
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0B0E14;
        color: #E0E0E0;
    }

    /* Force visibility for all labels and text */
    label, .stSlider p, .stSelectbox p, .stNumberInput p, .stMarkdown p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* Title Styling */
    .main-title {
        font-size: 50px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Subtitle Styling */
    .sub-title {
        text-align: center;
        color: #8B949E !important;
        font-size: 1.2rem !important;
        margin-bottom: 40px;
    }

    /* Section Headers */
    h3 {
        color: #00D2FF !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #1F2937;
        padding-bottom: 10px;
        margin-top: 10px !important;
    }

    /* Input Card Containers */
    [data-testid="stVerticalBlock"] > div:has(div.stSlider), 
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput),
    [data-testid="stVerticalBlock"] > div:has(div.stSelectbox) {
        background-color: #161B22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    /* Hover effect for better focus */
    [data-testid="stVerticalBlock"] > div:hover {
        
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 210, 255, 0.1);
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: bold;
        border: none;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
        color: white !important;
    }

    /* Result Box Container */
    .result-card {
        background: #1F2937;
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #30363D;
        text-align: center;
        margin-top: 30px;
    }

    /* Custom Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0B0E14;
        color: #4B5563;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #1F2937;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_taxi_model():
    try:
        return joblib.load("xgboost_taxi_model.pkl")
    except:
        return None

model = load_taxi_model()

# =========================
# HEADER SECTION
# =========================
st.markdown("<h1 class='main-title'>🚖 TAXI DEMAND ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predicting urban mobility demand using XGBoost Intelligence</p>", unsafe_allow_html=True)

# =========================
# MAIN INPUTS
# =========================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("###  Temporal Attributes")
    hour = st.slider("Hour of Day (24h format)", 0, 23, 12)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month of Year", 1, 12, 6)
    day_of_week = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

with col2:
    st.markdown("###  Location & Trip Data")
    pickup_lat = st.number_input("Pickup Latitude (e.g., 40.7128)", value=40.7128, format="%.4f")
    pickup_lon = st.number_input("Pickup Longitude (e.g., -74.0060)", value=-74.0060, format="%.4f")
    trip_dist = st.number_input("Trip Distance (km)", value=5.0)
    trip_dur = st.number_input("Trip Duration (min)", value=20.0)

st.write("##") # Spacer

# Additional parameters in a cleaner view
with st.expander("⚙️ Fine-Tune Parameters"):
    c3, c4 = st.columns(2)
    with c3:
        passengers = st.slider("Number of Passengers", 1, 6, 1)
    with c4:
        fare = st.number_input("Estimated Fare Amount ($)", value=15.0)

# =========================
# FEATURE ENGINEERING
# =========================
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
is_peak = 1 if hour in [7,8,9,17,18,19] else 0
speed = trip_dist / (trip_dur / 60 + 0.01) # Avoid division by zero

# =========================
# PREDICTION LOGIC
# =========================
st.write("##")
if st.button("✨ GENERATE PREDICTION"):
    # Prepare data for model
    features = pd.DataFrame([{
        'hour': hour, 'day': day, 'month': month, 'day_of_week': day_map[day_of_week],
        'is_peak': is_peak, 'pickup_latitude': round(pickup_lat, 2),
        'pickup_longitude': round(pickup_lon, 2), 'trip_distance': trip_dist,
        'trip_duration': trip_dur, 'speed': speed, 'passenger_count': passengers,
        'fare_amount': fare
    }])

    if model:
        pred = model.predict(features)
        result = int(np.expm1(pred[0]))
    else:
        # Fallback for display
        result = np.random.randint(5, 120)
        st.warning("⚠️ Model file 'xgboost_taxi_model.pkl' not found. Displaying demo result.")

    # Result Display Card
    st.markdown(f"""
        <div class="result-card">
            <h4 style="color: #00D2FF; margin-bottom: 10px; font-weight: 300;">PREDICTED BOOKING VOLUME</h4>
            <h1 style="font-size: 80px; margin: 0; color: white;">{result}</h1>
            <p style="color: #8B949E;">Estimated requests for this location and time</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("##")

    # Feedback indicators based on demand
  # Feedback indicators with forced high-visibility colors
    if result < 30:
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00FF00;">
                <span style="color: #00FF00 !important; font-size: 20px; font-weight: 900;">
                    🟢 LOW DEMAND: Drivers are likely idling. Fast pickup available.
                </span>
            </div>
        """, unsafe_allow_html=True)
        
    elif result < 75:
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: rgba(255, 255, 0, 0.1); border: 1px solid #FFFF00;">
                <span style="color: #FFFF00 !important; font-size: 20px; font-weight: 900;">
                    🟡 MODERATE DEMAND: Balanced market. Standard wait times.
                </span>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: rgba(255, 75, 75, 0.1); border: 1px solid #FF4B4B;">
                <span style="color: #FF4B4B !important; font-size: 20px; font-weight: 900;">
                    🔴 HIGH DEMAND: Significant congestion. Surge pricing likely in effect.
                </span>
            </div>
        """, unsafe_allow_html=True)
# =========================
# FOOTER
# =========================
st.markdown("""
    <div class="footer">
        DSBDA Mini Project | Optimized for Streamlit Cloud | © 2026
    </div>
""", unsafe_allow_html=True)