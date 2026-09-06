import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ElectroGuard AI",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("electrical_fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================================================
# DEMO USER STORE
# =========================================================
# NOTE: this is an in-memory demo store only — accounts reset every time the
# app restarts. Swap this for a real database / secrets-based auth (e.g.
# streamlit-authenticator, Supabase, Firebase) before using this in production.
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "admin123"
    }

# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"   # "login" or "signup"
if "auth_error" not in st.session_state:
    st.session_state.auth_error = ""
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a1e3d 0%, #0c2a54 45%, #081833 100%);
    background-attachment: fixed;
    color: #ffffff;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: repeating-linear-gradient(
        135deg,
        rgba(255,255,255,0.035) 0px,
        rgba(255,255,255,0.035) 2px,
        transparent 2px,
        transparent 70px
    );
    z-index: 0;
}

.block-container {
    max-width: 1450px;
    padding: 25px 55px 60px 55px;
    position: relative;
    z-index: 1;
}

header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ================= NAVBAR ================= */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 65px;
    border-bottom: 1px solid #172333;
    margin-bottom: 35px;
}
.logo-wrapper { display: flex; align-items: center; gap: 12px; }
.logo {
    width: 42px; height: 42px; border-radius: 10px;
    background: #1683ff; display: flex; justify-content: center;
    align-items: center; font-size: 21px;
}
.brand { font-size: 20px; font-weight: 800; }
.brand-small { font-size: 9px; color: #61748a; letter-spacing: 1.5px; }
.online { color: #30d98a; font-size: 12px; display: flex; align-items: center; gap: 7px; }
.online-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #30d98a; box-shadow: 0 0 10px #30d98a;
}

/* ================= LOGIN NAVBAR ================= */
.login-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 10px;
    margin-bottom: 60px;
}
.login-navbar .brand { font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }
.login-navlinks { display: flex; gap: 34px; align-items: center; }
.login-navlinks span {
    font-size: 13px; letter-spacing: 1px; color: #cfd9e6; font-weight: 600;
}
.login-navlinks .active {
    color: #ffffff;
    border-bottom: 2px solid #36a5ff;
    padding-bottom: 6px;
}

/* ================= LOGIN CARD ================= */
.login-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 26px;
    letter-spacing: 0.5px;
}
.login-subtext {
    text-align: center;
    color: #8498ac;
    font-size: 13px;
    margin-top: -16px;
    margin-bottom: 22px;
}
.forgot-link, .signup-text a {
    color: #4aa8ff;
    text-decoration: none;
    font-size: 12.5px;
}
.forgot-link:hover, .signup-text a:hover { text-decoration: underline; }
.signup-text {
    text-align: center;
    color: #7d90a4;
    font-size: 13px;
    margin-top: 18px;
}
.remember-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: -8px;
    margin-bottom: 6px;
}
.auth-error {
    background: rgba(255,67,67,0.08);
    border: 1px solid rgba(255,67,67,0.3);
    color: #ff8787;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 14px;
    text-align: center;
}
.auth-success {
    background: rgba(31,210,135,0.08);
    border: 1px solid rgba(31,210,135,0.3);
    color: #5be6b0;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 14px;
    text-align: center;
}

/* login card container (targets the bordered st.container) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0b1626 !important;
    border: 1px solid #1a2c40 !important;
    border-radius: 22px !important;
    box-shadow: 0 30px 80px rgba(0,0,0,0.45);
    padding: 10px 6px;
}

/* ================= HERO ================= */
.hero {
    min-height: 490px; border-radius: 24px; padding: 65px;
    position: relative; overflow: hidden;
    background: radial-gradient(circle at 80% 40%, rgba(19,133,255,0.18), transparent 32%),
                linear-gradient(135deg, #0a1a2d, #071321);
    border: 1px solid #17283b;
}
.grid {
    position: absolute; inset: 0; opacity: 0.13;
    background-image: linear-gradient(#3b76a5 1px, transparent 1px),
                       linear-gradient(90deg, #3b76a5 1px, transparent 1px);
    background-size: 50px 50px;
}
.hero-content { position: relative; z-index: 2; width: 58%; }
.tag {
    display: inline-block; padding: 8px 14px; border: 1px solid #1b6dab;
    border-radius: 30px; color: #42aaff; font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; background: rgba(20,130,255,0.07);
}
.hero-title { font-size: 58px; line-height: 1.04; font-weight: 800; letter-spacing: -2px; margin-top: 22px; }
.blue { color: #36a5ff; }
.hero-text { color: #8498ac; font-size: 16px; line-height: 1.8; max-width: 610px; margin-top: 20px; }

/* ================= MONITOR ================= */
.monitor {
    position: absolute; right: 55px; top: 65px; width: 390px; height: 355px;
    background: #081522; border: 1px solid #19354d; border-radius: 18px;
    padding: 22px; box-shadow: 0 25px 70px rgba(0,0,0,0.4);
}
.monitor-header { display: flex; justify-content: space-between; color: #8195a9; font-size: 11px; }
.live { color: #28d58a; }
.wave {
    margin-top: 25px; height: 155px;
    background: linear-gradient(#102c40 1px, transparent 1px),
                linear-gradient(90deg, #102c40 1px, transparent 1px);
    background-size: 30px 30px; border-radius: 10px;
}
.wave svg { width: 100%; height: 100%; }
.monitor-data { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 18px; }
.data { background: #0d1c2b; border-radius: 9px; padding: 12px; }
.data-label { color: #5f7489; font-size: 9px; }
.data-value { font-size: 17px; font-weight: 700; margin-top: 4px; }

/* ================= SECTIONS ================= */
.section { margin-top: 75px; }
.kicker { color: #2799ff; font-size: 10px; font-weight: 800; letter-spacing: 2px; }
.title { font-size: 34px; font-weight: 800; margin-top: 8px; }
.subtitle { color: #71869b; font-size: 14px; max-width: 700px; line-height: 1.7; margin-top: 8px; }

/* ================= STATS ================= */
.stat-card { background: #0b1624; border: 1px solid #172638; border-radius: 14px; padding: 23px; }
.stat-number { font-size: 30px; font-weight: 800; color: #ffffff; }
.stat-label { color: #63788d; font-size: 11px; margin-top: 5px; }

/* ================= PROCESS ================= */
.process { display: grid; grid-template-columns: repeat(4,1fr); margin-top: 35px; }
.process-card { padding: 25px; border-left: 1px solid #1a2b3d; min-height: 170px; }
.process-card:first-child { border-left: 2px solid #208fff; }
.number { color: #208fff; font-size: 11px; font-weight: 800; }
.process-title { font-size: 18px; font-weight: 700; margin-top: 15px; }
.process-text { color: #657a8e; font-size: 12px; line-height: 1.65; margin-top: 8px; }

/* ================= FAULT / GENERIC CARDS ================= */
.fault-card { background: #0b1624; border: 1px solid #172638; border-radius: 14px; padding: 25px; min-height: 145px; }
.fault-code { color: #278fff; font-size: 9px; font-weight: 800; letter-spacing: 1.5px; }
.fault-title { font-size: 18px; font-weight: 700; margin-top: 13px; }
.fault-text { color: #667c90; font-size: 11px; line-height: 1.6; margin-top: 7px; }

/* ================= DASHBOARD ================= */
.dashboard-header {
    padding: 30px;
    background: linear-gradient(135deg, #0d263e, #0a1a2c);
    border-radius: 18px; border: 1px solid #183249;
}
.dashboard-title { font-size: 30px; font-weight: 800; }
.dashboard-text { color: #7890a5; font-size: 13px; margin-top: 7px; }

/* ================= RESULT ================= */
.result-normal {
    padding: 30px; border-radius: 15px;
    background: rgba(31,210,135,0.07); border: 1px solid rgba(31,210,135,0.25);
}
.result-fault {
    padding: 30px; border-radius: 15px;
    background: rgba(255,67,67,0.07); border: 1px solid rgba(255,67,67,0.25);
}
.result-label { font-size: 9px; letter-spacing: 2px; color: #71879a; }
.result-value { font-size: 32px; font-weight: 800; margin-top: 8px; }
.result-info { color: #8296a9; font-size: 13px; margin-top: 8px; }

/* ================= INPUTS ================= */
label { color: #a7b8c8 !important; }
div[data-baseweb="input"] { background: #0c1927 !important; border-radius: 10px !important; }
input { color: white !important; }
div[data-baseweb="input"] > div { border-radius: 10px !important; border-color: #1c2f44 !important; }

/* primary buttons -> electric gradient */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1683ff, #2f5bff) !important;
    border: none !important;
    box-shadow: 0 10px 25px rgba(22,131,255,0.35);
}
.stButton > button[kind="secondary"] {
    background: #0c1927 !important;
    border: 1px solid #1c2f44 !important;
    color: #cfd9e6 !important;
}

/* ================= FOOTER ================= */
.footer {
    margin-top: 90px; padding-top: 25px; border-top: 1px solid #172333;
    text-align: center; color: #506478; font-size: 10px; letter-spacing: 1px;
}
.footer a { color: #4aa8ff; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# AUTH HELPERS
# =========================================================
def attempt_login(username: str, password: str) -> bool:
    return (
        username in st.session_state.users
        and st.session_state.users[username] == password
    )


def attempt_signup(username: str, password: str, confirm: str):
    if not username or not password:
        return False, "Please fill in both a username and a password."
    if username in st.session_state.users:
        return False, "That username is already taken."
    if password != confirm:
        return False, "Passwords do not match."
    st.session_state.users[username] = password
    return True, "Account created — you can log in now."


# =========================================================
# LOGIN / SIGN UP PAGE
# =========================================================
def render_auth_page():
    st.markdown("""
    <div class="login-navbar">
        <div class="brand">⚡ ElectroGuard AI</div>
        <div class="login-navlinks">
            <span>HOME</span>
            <span>ABOUT US</span>
            <span>CONTACT</span>
            <span class="active">LOG IN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1.15, 1])

    with mid:
        with st.container(border=True):
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.session_state.auth_mode == "login":
                st.markdown('<div class="login-title">Log in</div>', unsafe_allow_html=True)

                if st.session_state.auth_error:
                    st.markdown(
                        f'<div class="auth-error">{st.session_state.auth_error}</div>',
                        unsafe_allow_html=True
                    )

                username = st.text_input(
                    "Username", placeholder="Username", label_visibility="collapsed"
                )
                password = st.text_input(
                    "Password", type="password", placeholder="Password",
                    label_visibility="collapsed"
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.checkbox("Remember me", key="remember_me")
                with c2:
                    st.markdown(
                        '<div style="text-align:right; margin-top:8px;">'
                        '<a class="forgot-link" href="#">Forgot Password?</a></div>',
                        unsafe_allow_html=True
                    )

                if st.button("Log in", use_container_width=True, type="primary"):
                    if attempt_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.session_state.auth_error = ""
                        st.rerun()
                    else:
                        st.session_state.auth_error = "Incorrect username or password."
                        st.rerun()

                st.markdown(
                    '<div class="signup-text">Don\'t have an account? '
                    '<a href="#" id="signup-link">Sign up</a></div>',
                    unsafe_allow_html=True
                )
                if st.button("Create an account instead", use_container_width=True, type="secondary"):
                    st.session_state.auth_mode = "signup"
                    st.session_state.auth_error = ""
                    st.rerun()

                st.caption("Demo credentials — username: `admin`, password: `admin123`")

            else:
                st.markdown('<div class="login-title">Sign up</div>', unsafe_allow_html=True)

                if st.session_state.auth_error:
                    st.markdown(
                        f'<div class="auth-error">{st.session_state.auth_error}</div>',
                        unsafe_allow_html=True
                    )

                new_username = st.text_input(
                    "New username", placeholder="Choose a username", label_visibility="collapsed"
                )
                new_password = st.text_input(
                    "New password", type="password", placeholder="Choose a password",
                    label_visibility="collapsed"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password", placeholder="Confirm password",
                    label_visibility="collapsed"
                )

                if st.button("Sign up", use_container_width=True, type="primary"):
                    ok, message = attempt_signup(new_username, new_password, confirm_password)
                    if ok:
                        st.session_state.auth_mode = "login"
                        st.session_state.auth_error = ""
                        st.rerun()
                    else:
                        st.session_state.auth_error = message
                        st.rerun()

                st.markdown(
                    '<div class="signup-text">Already have an account? '
                    '<a href="#">Log in</a></div>',
                    unsafe_allow_html=True
                )
                if st.button("Back to log in", use_container_width=True, type="secondary"):
                    st.session_state.auth_mode = "login"
                    st.session_state.auth_error = ""
                    st.rerun()

    st.markdown(
        '<div class="footer">ElectroGuard AI &copy; 2026 &middot; '
        '<a href="https://github.com/avanthikamanoj900-arch/Ai-electricity-fault-prediction" target="_blank">'
        'View source on GitHub</a></div>',
        unsafe_allow_html=True
    )


# =========================================================
# GATE THE APP BEHIND LOGIN
# =========================================================
if not st.session_state.logged_in:
    render_auth_page()
    st.stop()

# =========================================================
# MAIN APP NAVBAR (post-login)
# =========================================================
st.markdown(f"""
<div class="navbar">
    <div class="logo-wrapper">
        <div class="logo">⚡</div>
        <div>
            <div class="brand">ElectroGuard AI</div>
            <div class="brand-small">INTELLIGENT POWER MONITORING</div>
        </div>
    </div>
    <div class="online">
        <div class="online-dot"></div>
        SYSTEM ONLINE &middot; {st.session_state.current_user}
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# NAVIGATION
# =========================================================
n1, n2, n3, n4, n5 = st.columns(5)
with n1:
    if st.button("Overview", use_container_width=True):
        st.session_state.page = "Overview"
        st.rerun()
with n2:
    if st.button("Fault Analysis", use_container_width=True):
        st.session_state.page = "Analysis"
        st.rerun()
with n3:
    if st.button("System Architecture", use_container_width=True):
        st.session_state.page = "System"
        st.rerun()
with n4:
    if st.button("About Project", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()
with n5:
    if st.button("Log out", use_container_width=True, type="secondary"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

# =========================================================
# OVERVIEW
# =========================================================
if st.session_state.page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="grid"></div>
        <div class="hero-content">
            <div class="tag">AI-BASED ELECTRICAL FAULT DETECTION</div>
            <div class="hero-title">
                Predict electrical faults
                <br>
                <span class="blue">before failure.</span>
            </div>
            <div class="hero-text">
                ElectroGuard AI analyzes voltage, current, frequency,
                power factor, temperature and THD using a trained
                machine-learning model to identify abnormal
                electrical conditions.
            </div>
        </div>
        <div class="monitor">
            <div class="monitor-header">
                <span>POWER SYSTEM MONITOR</span>
                <span class="live">&#9679; LIVE</span>
            </div>
            <div class="wave">
                <svg viewBox="0 0 360 155" preserveAspectRatio="none">
                    <polyline points="0,78 15,78 25,35 40,120 55,78 75,78 90,42 105,110 120,78 145,78 160,30 175,125 190,78 215,78 230,40 245,115 260,78 285,78 300,32 315,122 330,78 360,78"
                        fill="none" stroke="#229cff" stroke-width="3" />
                </svg>
            </div>
            <div class="monitor-data">
                <div class="data">
                    <div class="data-label">VOLTAGE</div>
                    <div class="data-value">230 V</div>
                </div>
                <div class="data">
                    <div class="data-label">CURRENT</div>
                    <div class="data-value">5.0 A</div>
                </div>
                <div class="data">
                    <div class="data-label">STATUS</div>
                    <div class="data-value">NORMAL</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("START ANALYSIS →", type="primary", use_container_width=True):
            st.session_state.page = "Analysis"
            st.rerun()
    with c2:
        if st.button("VIEW SYSTEM", use_container_width=True):
            st.session_state.page = "System"
            st.rerun()

    st.markdown("""
    <div class="section">
        <div class="kicker">MODEL PERFORMANCE</div>
        <div class="title">Machine-learning system overview</div>
        <div class="subtitle">
            A Random Forest classification model processes six
            electrical parameters to identify four operating conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    stats = [
        ("91.67%", "Prototype accuracy"),
        ("6", "Input parameters"),
        ("4", "Fault classes"),
        ("100", "Decision trees")
    ]
    for col, (num, text) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{num}</div>
                    <div class="stat-label">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="section">
        <div class="kicker">HOW IT WORKS</div>
        <div class="title">From electrical signal to AI decision</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="process">
        <div class="process-card">
            <div class="number">01</div>
            <div class="process-title">Measure</div>
            <div class="process-text">Electrical operating parameters are collected from the monitored system.</div>
        </div>
        <div class="process-card">
            <div class="number">02</div>
            <div class="process-title">Normalize</div>
            <div class="process-text">Measurements are transformed using the same preprocessing pipeline used during training.</div>
        </div>
        <div class="process-card">
            <div class="number">03</div>
            <div class="process-title">Classify</div>
            <div class="process-text">The Random Forest model evaluates the feature pattern and determines the operating condition.</div>
        </div>
        <div class="process-card">
            <div class="number">04</div>
            <div class="process-title">Report</div>
            <div class="process-text">The detected condition and model confidence are presented to the operator.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <div class="kicker">FAULT CLASSIFICATION</div>
        <div class="title">What ElectroGuard can detect</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    faults = [
        ("F-01", "Normal", "Operating parameters remain within the expected range."),
        ("F-02", "Overvoltage", "Voltage rises above the expected operating condition."),
        ("F-03", "Undervoltage", "Voltage falls below the expected operating condition."),
        ("F-04", "Overcurrent", "Current exceeds the expected operating condition.")
    ]
    for col, (code, title, text) in zip(cols, faults):
        with col:
            st.markdown(
                f"""
                <div class="fault-card">
                    <div class="fault-code">{code}</div>
                    <div class="fault-title">{title}</div>
                    <div class="fault-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# ANALYSIS
# =========================================================
elif st.session_state.page == "Analysis":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Electrical Fault Analysis</div>
        <div class="dashboard-text">
            Enter the measured electrical parameters below.
            The trained machine-learning model will classify
            the operating condition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="fault-card">
            <div class="fault-code">INPUT PARAMETERS</div>
            <div class="fault-title">Electrical measurements</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            voltage = st.number_input("Voltage (V)", 0.0, 500.0, 230.0, 1.0)
            current = st.number_input("Current (A)", 0.0, 500.0, 5.0, 0.5)
            frequency = st.number_input("Frequency (Hz)", 0.0, 100.0, 50.0, 0.1)
        with c2:
            power_factor = st.number_input("Power Factor", 0.0, 1.0, 0.95, 0.01)
            temperature = st.number_input("Temperature (°C)", -20.0, 150.0, 30.0, 1.0)
            thd = st.number_input("THD (%)", 0.0, 100.0, 2.0, 0.5)

        if st.button("RUN AI FAULT ANALYSIS", type="primary", use_container_width=True):
            data = np.array([[voltage, current, frequency, power_factor, temperature, thd]])
            scaled_data = scaler.transform(data)
            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]
            confidence = float(np.max(probabilities) * 100)

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.rerun()

    with col2:
        st.markdown("""
        <div class="fault-card">
            <div class="fault-code">AI OUTPUT</div>
            <div class="fault-title">Prediction result</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.prediction is None:
            st.info("Enter the electrical parameters and run the AI analysis.")
        else:
            # NOTE: adjust this label map to match the exact class order your
            # model was trained/encoded with (check train_model.py).
            label_map = {
                0: "Normal",
                1: "Overvoltage",
                2: "Undervoltage",
                3: "Overcurrent"
            }
            try:
                pred_label = label_map.get(int(st.session_state.prediction), str(st.session_state.prediction))
            except (TypeError, ValueError):
                pred_label = str(st.session_state.prediction)

            css_class = "result-normal" if pred_label == "Normal" else "result-fault"

            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="result-label">DETECTED CONDITION</div>
                    <div class="result-value">{pred_label}</div>
                    <div class="result-info">
                        Model confidence: {st.session_state.confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# SYSTEM ARCHITECTURE
# =========================================================
elif st.session_state.page == "System":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">System Architecture</div>
        <div class="dashboard-text">
            How ElectroGuard AI turns raw electrical readings into a
            fault classification, end to end.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="process">
        <div class="process-card">
            <div class="number">INPUT</div>
            <div class="process-title">Electrical parameters</div>
            <div class="process-text">Voltage, current, frequency, power factor, temperature and THD, entered manually or streamed from sensors.</div>
        </div>
        <div class="process-card">
            <div class="number">PREP</div>
            <div class="process-title">scaler.pkl</div>
            <div class="process-text">A saved StandardScaler applies the same normalization used during training before inference.</div>
        </div>
        <div class="process-card">
            <div class="number">MODEL</div>
            <div class="process-title">electrical_fault_model.pkl</div>
            <div class="process-text">A Random Forest classifier (trained in train_model.py) predicts the operating condition and a confidence score.</div>
        </div>
        <div class="process-card">
            <div class="number">UI</div>
            <div class="process-title">Streamlit app</div>
            <div class="process-text">app.py serves the login gate, dashboard and results as a single-page Streamlit application.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    cols = st.columns(3)
    stack = [
        ("Frontend", "Streamlit, custom CSS for the electric-themed dashboard and login screen."),
        ("Modeling", "scikit-learn Random Forest classifier + StandardScaler, serialized with joblib."),
        ("Auth", "Lightweight session-based login/sign-up gate in front of the dashboard.")
    ]
    for col, (title, text) in zip(cols, stack):
        with col:
            st.markdown(
                f"""
                <div class="fault-card">
                    <div class="fault-code">STACK</div>
                    <div class="fault-title">{title}</div>
                    <div class="fault-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# ABOUT PROJECT
# =========================================================
elif st.session_state.page == "About":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">About the Project</div>
        <div class="dashboard-text">
            AI-Based Electrical Fault Prediction System — an ML system that
            classifies electrical operating conditions from live parameters.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="fault-card">
        <div class="fault-code">PROJECT</div>
        <div class="fault-title">What it does</div>
        <div class="fault-text">
            This system takes voltage, current, frequency, power factor,
            temperature and total harmonic distortion (THD) readings and
            uses a trained Random Forest model to flag whether the system
            is Normal or experiencing an Overvoltage, Undervoltage or
            Overcurrent fault — enabling earlier detection and preventive
            maintenance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.link_button(
        "View the repository on GitHub",
        "https://github.com/avanthikamanoj900-arch/Ai-electricity-fault-prediction",
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    ElectroGuard AI &copy; 2026 &middot; AI-Based Electrical Fault Prediction System
</div>
""", unsafe_allow_html=True)
