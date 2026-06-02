import streamlit as st
import cv2
import os
import joblib
import pandas as pd
import numpy as np
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from streamlit_option_menu import option_menu

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="FaceAttend Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================
# PREMIUM UI — Deep Space / Bioluminescent Theme
# ====================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:        #040d1a;
    --bg-card:        #071428;
    --bg-card2:       #0a1c35;
    --accent-cyan:    #00f5d4;
    --accent-blue:    #0ea5e9;
    --accent-violet:  #818cf8;
    --accent-green:   #22d3a5;
    --text-primary:   #e2f4ff;
    --text-secondary: #7da8c8;
    --border:         rgba(0,245,212,0.15);
    --glow-cyan:      0 0 20px rgba(0,245,212,0.25);
    --glow-blue:      0 0 20px rgba(14,165,233,0.3);
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text-primary) !important;
}

/* Animated deep-space mesh background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,245,212,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 80%, rgba(14,165,233,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(129,140,248,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Grid overlay */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,245,212,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,212,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stVerticalBlock"] { position: relative; z-index: 1; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

p, span, div, label, li {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Hero Header ── */
.hero-header {
    padding: 48px 0 32px;
    text-align: center;
    position: relative;
}
.hero-header .badge {
    display: inline-block;
    background: rgba(0,245,212,0.1);
    border: 1px solid rgba(0,245,212,0.3);
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.2em;
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 16px;
}
.hero-header h1 {
    font-size: clamp(32px, 5vw, 64px) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #e2f4ff 0%, var(--accent-cyan) 50%, var(--accent-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px !important;
    line-height: 1.1 !important;
}
.hero-header p {
    color: var(--text-secondary) !important;
    font-size: 15px !important;
    max-width: 500px;
    margin: 0 auto;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.6;
}
.card:hover {
    border-color: rgba(0,245,212,0.3);
    box-shadow: var(--glow-cyan);
}

.card-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 8px !important;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
}
.metric-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--accent-cyan) !important;
    display: block;
    margin-bottom: 4px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.15em;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

/* ── Developer Card ── */
.dev-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.dev-avatar {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.dev-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
.dev-links a {
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-decoration: none;
    margin-right: 16px;
}
.dev-links a:hover { text-decoration: underline; }

/* ── Navigation ── */
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, rgba(0,245,212,0.15), rgba(14,165,233,0.15)) !important;
    border: 1px solid rgba(0,245,212,0.35) !important;
    color: var(--accent-cyan) !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,245,212,0.25), rgba(14,165,233,0.25)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)) !important;
    border-radius: 100px;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 100px;
}

/* ── Alerts ── */
.stSuccess, .stInfo, .stError, .stWarning {
    border-radius: 12px !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stSuccess { background: rgba(34,211,165,0.12) !important; border-left: 3px solid var(--accent-green) !important; }
.stInfo    { background: rgba(14,165,233,0.12) !important; border-left: 3px solid var(--accent-blue) !important; }
.stError   { background: rgba(239,68,68,0.12) !important;  border-left: 3px solid #ef4444 !important; }
.stWarning { background: rgba(245,158,11,0.12) !important; border-left: 3px solid #f59e0b !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent-cyan) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}
.dvn-scroller { background: var(--bg-card) !important; }

/* ── Section Heading ── */
.section-heading {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin-bottom: 4px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-sub {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text-secondary) !important;
    margin-bottom: 24px !important;
    letter-spacing: 0.05em;
}

/* ── Status Dot ── */
.dot-online {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(0,245,212,0.3); border-radius: 3px; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(34,211,165,0.15), rgba(0,245,212,0.1)) !important;
    border: 1px solid rgba(34,211,165,0.4) !important;
    color: var(--accent-green) !important;
}
</style>
""", unsafe_allow_html=True)

# ====================================
# HERO HEADER
# ====================================

st.markdown("""
<div class="hero-header">
    <div class="badge">⬡ AI-POWERED BIOMETRIC SYSTEM v2.0</div>
    <h1>FaceAttend Pro</h1>
    <p>Real-time face recognition · gender detection · automated attendance tracking</p>
</div>
""", unsafe_allow_html=True)

# ====================================
# DEVELOPER CARD
# ====================================

st.markdown("""
<div class="dev-card">
    <div class="dev-avatar">👨‍💻</div>
    <div>
        <div class="dev-name">Chandan Kumar Sah</div>
        <div class="dev-links">
            <a href="https://github.com/ChankumarSah" target="_blank">⌥ GitHub</a>
            <a href="https://www.linkedin.com/in/chandan-kumar-sah-752803387" target="_blank">⌥ LinkedIn</a>
        </div>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <span class="dot-online"></span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#22d3a5; margin-left:6px;">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# NAVIGATION
# ====================================

selected = option_menu(
    menu_title=None,
    options=["Recognition", "Upload Image", "Register Person", "Train Model", "Attendance"],
    icons=["camera-video-fill", "image-fill", "person-plus-fill", "cpu-fill", "table"],
    orientation="horizontal",
    styles={
        "container": {
            "background-color": "#071428",
            "border": "1px solid rgba(0,245,212,0.15)",
            "border-radius": "14px",
            "padding": "6px",
            "margin-bottom": "28px"
        },
        "icon": {"color": "#00f5d4", "font-size": "16px"},
        "nav-link": {
            "font-size": "13px",
            "font-family": "'Syne', sans-serif",
            "font-weight": "600",
            "color": "#7da8c8",
            "border-radius": "10px",
            "padding": "10px 18px"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, rgba(0,245,212,0.2), rgba(14,165,233,0.2))",
            "color": "#00f5d4",
            "border": "1px solid rgba(0,245,212,0.3)"
        }
    }
)

# ====================================
# CREATE FOLDERS
# ====================================

for folder in ["dataset", "models", "unknown_faces"]:
    os.makedirs(folder, exist_ok=True)

# ====================================
# ATTENDANCE FILE
# ====================================

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)

# ====================================
# LOAD MODELS
# ====================================

@st.cache_resource
def load_models():
    try:
        name_model   = joblib.load("models/name_model_clf.pkl")
        name_pca     = joblib.load("models/name_pca_clf.pkl")
        gender_model = joblib.load("models/gender_model.pkl")
        gender_pca   = joblib.load("models/gender_clf_pca.pkl")
        return name_model, name_pca, gender_model, gender_pca
    except Exception:
        return None, None, None, None

name_model, name_pca, gender_model, gender_pca = load_models()

# ====================================
# FACE DETECTOR
# ====================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ====================================
# ATTENDANCE FUNCTION
# ====================================

def mark_attendance(name):
    df = pd.read_csv("attendance.csv")
    date = time.strftime("%d-%m-%Y")
    current_time = time.strftime("%H:%M:%S")
    already = ((df["Name"] == name) & (df["Date"] == date)).any()
    if not already:
        df.loc[len(df)] = [name, date, current_time]
        df.to_csv("attendance.csv", index=False)

# ====================================
# STATUS BAR
# ====================================

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📁 Dataset", "✅ Ready")
with col2:
    st.metric("🧠 Models", "✅ Loaded" if name_model is not None else "❌ Not Found")
with col3:
    st.metric("📋 Attendance", "✅ Ready")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ====================================
# PREDICT FACE HELPER
# ====================================

def predict_face(face_bgr):
    """
    Returns (label, color, name) for a face crop (BGR numpy array).
    BUG FIX: safely handles PCA shape mismatch and missing gender model.
    """
    face_resized = cv2.resize(face_bgr, (150, 150))
    face_gray    = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_flat    = face_gray.flatten() / 255.0

    # ── Name prediction ──
    expected_features = name_pca.n_features_in_
    if face_flat.shape[0] != expected_features:
        return "Shape Mismatch", (255, 165, 0), None

    face_pca    = name_pca.transform([face_flat])
    pred_name   = name_model.predict(face_pca)[0]
    confidence  = float(np.max(name_model.predict_proba(face_pca)))

    # ── Gender prediction (optional) ──
    pred_gender = ""
    if gender_model is not None and gender_pca is not None:
        try:
            face_g_pca  = gender_pca.transform([face_flat])
            pred_gender = gender_model.predict(face_g_pca)[0]
        except Exception:
            pred_gender = ""

    # ── Threshold ──
    if confidence < 0.85:
        label = f"Unknown  {confidence:.2f}"
        color = (0, 80, 255)
        return label, color, None
    else:
        parts = [pred_name]
        if pred_gender:
            parts.append(pred_gender)
        parts.append(f"{confidence:.2f}")
        label = " | ".join(parts)
        color = (0, 220, 120)
        return label, color, pred_name

# ====================================
# RECOGNITION PAGE
# ====================================

if selected == "Recognition":
    st.markdown("""
        <div class='section-heading'>🎥 Live Recognition</div>
        <div class='section-sub'>WEBCAM · REAL-TIME FACE IDENTIFICATION · AUTO ATTENDANCE</div>
    """, unsafe_allow_html=True)

    if name_model is None:
        st.error("⚠️ Models not found. Please train the model first on the **Train Model** page.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶  Start Camera")
        with col2:
            stop_btn  = st.button("⏹  Stop Camera")

        if "camera_on" not in st.session_state:
            st.session_state.camera_on = False

        if start_btn:
            st.session_state.camera_on = True
        if stop_btn:
            st.session_state.camera_on = False

        frame_ph = st.empty()
        info_ph  = st.empty()

        if st.session_state.camera_on:
            # BUG FIX: wrap in a loop so camera stays live; use STOP button to break
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not open webcam. Check camera permissions.")
                st.session_state.camera_on = False
            else:
                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        info_ph.warning("⚠️ Camera read failed. Retrying…")
                        time.sleep(0.05)
                        continue

                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    info_ph.info(f"👁  Faces Detected: **{len(faces)}**")

                    for (x, y, w, h) in faces:
                        label, color, name = predict_face(frame[y:y+h, x:x+w])
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        # Background bar behind text for readability
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
                        cv2.putText(frame, label, (x + 4, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                        if name:
                            mark_attendance(name)

                    frame_ph.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True
                    )
                    time.sleep(0.03)  # ~30 fps cap

                cap.release()
                frame_ph.empty()
                info_ph.success("✅ Camera stopped.")

# ====================================
# UPLOAD IMAGE PAGE
# ====================================

elif selected == "Upload Image":
    st.markdown("""
        <div class='section-heading'>📤 Image Prediction</div>
        <div class='section-sub'>UPLOAD · DETECT · IDENTIFY</div>
    """, unsafe_allow_html=True)

    if name_model is None:
        st.error("⚠️ Models not found. Please train the model first.")
    else:
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption="📎 Uploaded Image",
                    use_container_width=True
                )

            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>🔍 Run Prediction</div>", unsafe_allow_html=True)
                if st.button("Predict Faces in Image"):
                    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(faces) == 0:
                        st.error("No face detected in the image.")
                    else:
                        st.info(f"👁  Faces detected: **{len(faces)}**")
                        result_img = img.copy()
                        for (x, y, w, h) in faces:
                            label, color, name = predict_face(result_img[y:y+h, x:x+w])
                            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            cv2.rectangle(result_img, (x, y - th - 14), (x + tw + 8, y), color, -1)
                            cv2.putText(result_img, label, (x + 4, y - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                            if name:
                                mark_attendance(name)
                                st.success(f"✅ Attendance marked for **{name}**")

                        st.image(
                            cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                            caption="🧬 Prediction Result",
                            use_container_width=True
                        )
                st.markdown("</div>", unsafe_allow_html=True)

# ====================================
# REGISTER PERSON PAGE
# ====================================

elif selected == "Register Person":
    st.markdown("""
        <div class='section-heading'>➕ Register Person</div>
        <div class='section-sub'>CAPTURE FACE DATASET · MIN 100 IMAGES RECOMMENDED</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📝 Person Details</div>", unsafe_allow_html=True)
        person_name   = st.text_input("Full Name", placeholder="e.g. Chandan Kumar Sah")
        capture_count = st.slider("Images to Capture", min_value=100, max_value=500, value=300, step=50)
        st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace; font-size:11px; color:#7da8c8; margin-top:8px;'>
            ℹ️ More images = better accuracy. 300 is optimal.
            </div>
        """, unsafe_allow_html=True)
        capture_btn = st.button("📸  Start Capture")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        preview_ph = st.empty()
        counter_ph = st.empty()
        prog_ph    = st.empty()

    if capture_btn:
        if not person_name.strip():
            st.error("Please enter the person's name.")
        else:
            save_path = os.path.join("dataset", person_name.strip())
            os.makedirs(save_path, exist_ok=True)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam.")
            else:
                count = 0
                while count < capture_count:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    for (x, y, w, h) in faces:
                        face = cv2.resize(frame[y:y+h, x:x+w], (150, 150))
                        cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), face)
                        count += 1
                        prog_ph.progress(count / capture_count)
                        counter_ph.markdown(
                            f"<div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
                            f"color:#00f5d4;text-align:center;'>📸 {count} / {capture_count}</div>",
                            unsafe_allow_html=True
                        )
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 120), 2)
                        if count >= capture_count:
                            break

                    preview_ph.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True
                    )

                cap.release()
                st.success(f"✅ **{person_name}** registered with {count} images!")
                st.balloons()
                st.info("👉 Now go to **Train Model** to update the model.")

# ====================================
# TRAIN MODEL PAGE
# ====================================

elif selected == "Train Model":
    st.markdown("""
        <div class='section-heading'>🧠 Train Model</div>
        <div class='section-sub'>PCA + LOGISTIC REGRESSION · FACE RECOGNITION ENGINE</div>
    """, unsafe_allow_html=True)

    # BUG FIX: handle empty dataset folder gracefully
    dataset_path = "dataset"
    people = [
        p for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))
    ] if os.path.exists(dataset_path) else []

    total_images = sum(
        len(os.listdir(os.path.join(dataset_path, p)))
        for p in people
    ) if people else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Registered Persons", len(people))
    with col2:
        st.metric("🖼  Total Images", total_images)
    with col3:
        st.metric("📐 PCA Variance", "99.9%")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if len(people) == 0:
        st.warning("No persons registered yet. Please register at least one person first.")
    elif len(people) < 2:
        st.warning("⚠️ At least **2 persons** are required to train the classifier.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
            <div class='card-title'>🚀 Ready to Train</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:12px;color:#7da8c8;margin-bottom:16px;'>
            This will build a PCA + Logistic Regression model from your dataset.
            Existing model files will be overwritten.
            </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Train Now"):
            with st.spinner("Training in progress…"):

                X, y = [], []
                progress = st.progress(0)

                for idx, person in enumerate(people):
                    person_folder = os.path.join(dataset_path, person)
                    for img_name in os.listdir(person_folder):
                        img_path = os.path.join(person_folder, img_name)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.resize(img, (150, 150))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        X.append(img.flatten() / 255.0)
                        y.append(person)
                    # BUG FIX: safe progress update
                    progress.progress((idx + 1) / len(people))

                X = np.array(X)
                y = np.array(y)

                st.info(f"📊 Training samples: **{len(X)}**  |  Classes: **{len(np.unique(y))}**")

                pca   = PCA(0.999)
                X_pca = pca.fit_transform(X)
                st.success(f"✅ PCA reduced to **{X_pca.shape[1]}** components (from {X.shape[1]})")

                model = LogisticRegression(max_iter=5000)
                model.fit(X_pca, y)

                joblib.dump(model, "models/name_model_clf.pkl")
                joblib.dump(pca,   "models/name_pca_clf.pkl")

            st.success("🎉 Model trained and saved successfully!")
            st.balloons()
            st.info("🔁 Restart or reload the app to use the updated model.")

        st.markdown("</div>", unsafe_allow_html=True)

# ====================================
# ATTENDANCE PAGE
# ====================================

elif selected == "Attendance":
    st.markdown("""
        <div class='section-heading'>📋 Attendance Dashboard</div>
        <div class='section-sub'>RECORDS · EXPORT · MANAGE</div>
    """, unsafe_allow_html=True)

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")

        today = time.strftime("%d-%m-%Y")
        today_df = df[df["Date"] == today] if len(df) > 0 else pd.DataFrame()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📌 Total Records",   len(df))
        with col2:
            st.metric("👥 Unique Persons",  df["Name"].nunique() if len(df) > 0 else 0)
        with col3:
            st.metric("📅 Today Present",   len(today_df))
        with col4:
            st.metric("🗓  Today's Date",    today)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Filter
        col1, col2 = st.columns([2, 1])
        with col1:
            filter_name = st.text_input("🔍 Filter by Name", placeholder="Leave blank to show all")
        with col2:
            filter_date = st.text_input("📅 Filter by Date (DD-MM-YYYY)", placeholder="e.g. 01-06-2025")

        display_df = df.copy()
        if filter_name.strip():
            display_df = display_df[display_df["Name"].str.contains(filter_name.strip(), case=False, na=False)]
        if filter_date.strip():
            display_df = display_df[display_df["Date"] == filter_date.strip()]

        st.dataframe(
            display_df.reset_index(drop=True),
            use_container_width=True,
            height=420
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with open("attendance.csv", "rb") as f:
                st.download_button(
                    label="⬇  Download Attendance CSV",
                    data=f,
                    file_name="attendance.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("🗑  Clear All Attendance"):
                pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)
                st.success("Attendance cleared.")
                st.rerun()
    else:
        st.warning("No attendance file found.")
