import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ElectroGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("electrical_fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# CUSTOM WEBSITE STYLE
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background:
        radial-gradient(circle at 85% 10%, rgba(20,115,255,0.12), transparent 30%),
        radial-gradient(circle at 10% 80%, rgba(0,210,255,0.06), transparent 25%),
        #07111f;
    color: #ffffff;
}

.block-container {
    max-width: 1400px;
    padding: 1.2rem 4rem 4rem 4rem;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ---------- NAVBAR ---------- */

.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 0 25px 0;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 13px;
}

.logo {
    width: 43px;
    height: 43px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1e8cff, #00c8ff);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    font-weight: 900;
    color: white;
    box-shadow: 0 0 25px rgba(0,160,255,0.25);
}

.logo-text {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.logo-sub {
    font-size: 9px;
    color: #6f8198;
    letter-spacing: 1.4px;
    margin-top: 2px;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 8px 14px;
    color: #a8b7c8;
    font-size: 12px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22d58a;
    box-shadow: 0 0 10px #22d58a;
}


/* ---------- HERO ---------- */

.hero {
    min-height: 540px;
    border-radius: 28px;
    background:
        linear-gradient(
            110deg,
            rgba(8,28,50,0.98),
            rgba(9,39,70,0.93)
        );
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
    padding: 75px 70px;
}

.hero-grid {
    position: absolute;
    inset: 0;
    opacity: 0.18;
    background-image:
        linear-gradient(rgba(60,150,220,0.15) 1px, transparent 1px),
        linear-gradient(90deg, rgba(60,150,220,0.15) 1px, transparent 1px);
    background-size: 55px 55px;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 720px;
}

.eyebrow {
    display: inline-flex;
    padding: 7px 13px;
    border: 1px solid rgba(57,169,255,0.3);
    border-radius: 20px;
    background: rgba(30,140,255,0.08);
    color: #54b7ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.hero-title {
    font-size: 60px;
    line-height: 1.03;
    letter-spacing: -2.8px;
    font-weight: 850;
    margin-top: 24px;
}

.hero-title span {
    background: linear-gradient(90deg, #ffffff, #55b9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #91a6bb;
    font-size: 17px;
    line-height: 1.75;
    max-width: 620px;
    margin-top: 22px;
}

.hero-panel {
    position: absolute;
    right: 60px;
    top: 85px;
    width: 360px;
    height: 365px;
    background: rgba(5,18,32,0.72);
    border: 1px solid rgba(75,170,255,0.16);
    border-radius: 20px;
    padding: 23px;
    z-index: 3;
    backdrop-filter: blur(12px);
    box-shadow: 0 25px 80px rgba(0,0,0,0.35);
}

.panel-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.panel-title {
    font-size: 12px;
    color: #9cb1c5;
    letter-spacing: 1px;
}

.live {
    color: #28d991;
    font-size: 10px;
    font-weight: 700;
}

.wave-area {
    height: 150px;
    border-radius: 12px;
    background:
        linear-gradient(rgba(40,130,200,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(40,130,200,0.07) 1px, transparent 1px);
    background-size: 30px 30px;
    padding: 10px;
}

.wave {
    width: 100%;
    height: 100%;
}

.readings {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.reading {
    flex: 1;
    padding: 13px;
    background: rgba(255,255,255,0.035);
    border-radius: 10px;
}

.reading-label {
    font-size: 9px;
    color: #72869a;
    text-transform: uppercase;
}

.reading-value {
    font-size: 18px;
    font-weight: 750;
    margin-top: 5px;
}


/* ---------- BUTTONS ---------- */

.stButton > button {
    border-radius: 10px !important;
    height: 46px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #147fff, #00aeea) !important;
    color: white !important;
    border: none !important;
}


/* ---------- SECTION ---------- */

.section {
    margin-top: 80px;
}

.section-kicker {
    color: #3fa9ff;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 800;
    margin-bottom: 12px;
}

.section-title {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
}

.section-description {
    color: #71869b;
    max-width: 680px;
    line-height: 1.7;
    margin-top: 10px;
}


/* ---------- STATISTICS ---------- */

.stat {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 25px;
    min-height: 115px;
}

.stat-number {
    font-size: 30px;
    font-weight: 850;
}

.stat-label {
    color: #71869b;
    font-size: 12px;
    margin-top: 5px;
}


/* ---------- PIPELINE ---------- */

.pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    margin-top: 35px;
    border-top: 1px solid rgba(255,255,255,0.09);
}

.step {
    padding: 30px 25px;
    border-right: 1px solid rgba(255,255,255,0.07);
    position: relative;
}

.step:last-child {
    border-right: none;
}

.step-number {
    color: #319eff;
    font-size: 11px;
    font-weight: 800;
}

.step-title {
    font-size: 19px;
    font-weight: 750;
    margin-top: 18px;
}

.step-description {
    color: #71869b;
    font-size: 12px;
    line-height: 1.65;
    margin-top: 9px;
}


/* ---------- FAULT TYPES ---------- */

.fault {
    padding: 25px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 15px;
    min-height: 145px;
}

.fault-code {
    color: #3fa9ff;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 800;
}

.fault-title {
    font-size: 19px;
    font-weight: 750;
    margin-top: 12px;
}

.fault-description {
    color: #71869b;
    font-size: 12px;
    line-height: 1.6;
    margin-top: 7px;
}


/* ---------- DASHBOARD ---------- */

.dashboard {
    background: linear-gradient(135deg, #0b2038, #092b4d);
    border-radius: 20px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.07);
}

.dashboard-title {
    font-size: 32px;
    font-weight: 800;
}

.dashboard-description {
    color: #7f96aa;
    margin-top: 7px;
}


/* ---------- FORM ---------- */

.form-panel {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 30px;
}

.form-title {
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 25px;
}


/* ---------- RESULT ---------- */

.result {
    margin-top: 30px;
    padding: 35px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

.result-normal {
    background: rgba(25,190,120,0.08);
    border-color: rgba(25,210,140,0.2);
}

.result-fault {
    background: rgba(255,75,75,0.08);
    border-color: rgba(255,75,75,0.2);
}

.result-label {
    font-size: 10px;
    color: #7e95aa;
    letter-spacing: 1.5px;
}

.result-value {
    font-size: 34px;
    font-weight: 850;
    margin-top: 8px;
}

.result-confidence {
    color: #8ca1b5;
    margin-top: 8px;
}


/* ---------- FOOTER ---------- */

.footer {
    margin-top: 100px;
    padding-top: 25px;
    border-top: 1px solid rgba(255,255,255,0.07);
    text-align: center;
    color: #52677b;
    font-size: 11px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# NAVBAR
# ============================================================

st.markdown("""
<div class="nav">

    <div class="logo-area">

        <div class="logo">⚡</div>

        <div>
            <div class="logo-text">ElectroGuard AI</div>
            <div class="logo-sub">
                INTELLIGENT ELECTRICAL MONITORING
            </div>
        </div>

    </div>

    <div class="status">
        <div class="status-dot"></div>
        ML SYSTEM ONLINE
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION BUTTONS
# ============================================================

nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])

with nav1:
    if st.button("HOME", use_container_width=True):
        st.session_state.page = "Home"

with nav2:
    if st.button("FAULT ANALYSIS", use_container_width=True):
        st.session_state.page = "Analysis"

with nav3:
    if st.button("SYSTEM", use_container_width=True):
        st.session_state.page = "System"

with nav4:
    if st.button("ABOUT", use_container_width=True):
        st.session_state.page = "About"


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-grid"></div>

        <div class="hero-content">

            <div class="eyebrow">
                AI-POWERED ELECTRICAL FAULT DETECTION
            </div>

            <div class="hero-title">
                Predict electrical faults
                <span>before failure.</span>
            </div>

            <div class="hero-description">
                ElectroGuard AI uses machine learning to analyze
                electrical operating parameters and identify abnormal
                conditions before they develop into serious system faults.
            </div>

        </div>


        <div class="hero-panel">

            <div class="panel-top">

                <div class="panel-title">
                    POWER SYSTEM MONITOR
                </div>

                <div class="live">
                    ● LIVE
                </div>

            </div>


            <div class="wave-area">

                <svg class="wave"
                     viewBox="0 0 330 130"
                     preserveAspectRatio="none">

                    <polyline
                        points="
                        0,65
                        15,65
                        25,30
                        38,100
                        52,65
                        70,65
                        85,40
                        98,90
                        112,65
                        130,65
                        145,25
                        158,105
                        172,65
                        190,65
                        205,38
                        218,92
                        232,65
                        250,65
                        265,28
                        278,100
                        292,65
                        310,65
                        330,65"
                        fill="none"
                        stroke="#27a9ff"
                        stroke-width="3"
                    />

                </svg>

            </div>


            <div class="readings">

                <div class="reading">
                    <div class="reading-label">Voltage</div>
                    <div class="reading-value">230 V</div>
                </div>

                <div class="reading">
                    <div class="reading-label">Current</div>
                    <div class="reading-value">5.0 A</div>
                </div>

                <div class="reading">
                    <div class="reading-label">Status</div>
                    <div class="reading-value">Normal</div>
                </div>

            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # CTA
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.3, 1, 1.3])

    with c1:
        if st.button(
            "START FAULT ANALYSIS",
            type="primary",
            use_container_width=True
        ):
            st.session_state.page = "Analysis"
            st.rerun()

    with c2:
        if st.button(
            "VIEW SYSTEM",
            use_container_width=True
        ):
            st.session_state.page = "System"
            st.rerun()


    # ========================================================
    # STATISTICS
    # ========================================================

    st.markdown("""
    <div class="section">

        <div class="section-kicker">
            MODEL OVERVIEW
        </div>

        <div class="section-title">
            Built around six electrical parameters
        </div>

        <div class="section-description">
            The system evaluates multiple operating conditions
            simultaneously instead of relying on a single threshold.
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

    for col, (number, label) in zip(cols, stats):

        with col:

            st.markdown(
                f"""
                <div class="stat">

                    <div class="stat-number">
                        {number}
                    </div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # HOW SYSTEM WORKS
    # ========================================================

    st.markdown("""
    <div class="section">

        <div class="section-kicker">
            DETECTION PIPELINE
        </div>

        <div class="section-title">
            From measurement to prediction
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline">

        <div class="step">
            <div class="step-number">01</div>
            <div class="step-title">Measure</div>
            <div class="step-description">
                Electrical parameters are collected from
                the monitored system.
            </div>
        </div>

        <div class="step">
            <div class="step-number">02</div>
            <div class="step-title">Process</div>
            <div class="step-description">
                Measurements are normalized using
                the trained preprocessing pipeline.
            </div>
        </div>

        <div class="step">
            <div class="step-number">03</div>
            <div class="step-title">Predict</div>
            <div class="step-description">
                The Random Forest classifier evaluates
                the operating condition.
            </div>
        </div>

        <div class="step">
            <div class="step-number">04</div>
            <div class="step-title">Respond</div>
            <div class="step-description">
                The detected condition and confidence
                are presented to the operator.
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # FAULT TYPES
    # ========================================================

    st.markdown("""
    <div class="section">

        <div class="section-kicker">
            FAULT CLASSIFICATION
        </div>

        <div class="section-title">
            Conditions the model can identify
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fault_cols = st.columns(4)

    faults = [
        ("F-01", "Normal", "Operating parameters remain within the expected range."),
        ("F-02", "Overvoltage", "Voltage rises above the expected operating condition."),
        ("F-03", "Undervoltage", "Voltage falls below the expected operating condition."),
        ("F-04", "Overcurrent", "Current rises beyond the expected operating condition.")
    ]

    for col, (code, title, description) in zip(fault_cols, faults):

        with col:

            st.markdown(
                f"""
                <div class="fault">

                    <div class="fault-code">
                        {code}
                    </div>

                    <div class="fault-title">
                        {title}
                    </div>

                    <div class="fault-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FAULT ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Analysis":

    st.markdown("""
    <div class="dashboard">

        <div class="dashboard-title">
            Electrical Fault Analysis
        </div>

        <div class="dashboard-description">
            Enter measured electrical parameters and run the trained
            machine-learning model.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="form-panel">

        <div class="form-title">
            System measurements
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0,
            max_value=500.0,
            value=230.0,
            step=1.0
        )

        frequency = st.number_input(
            "Frequency (Hz)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )

    with col2:

        current = st.number_input(
            "Current (A)",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=0.5
        )

        power_factor = st.number_input(
            "Power Factor",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01
        )

    with col3:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=150.0,
            value=30.0,
            step=1.0
        )

        thd = st.number_input(
            "THD (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5
        )


    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "RUN AI ANALYSIS",
        type="primary",
        use_container_width=True
    ):

        input_data = np.array([[
            voltage,
            current,
            frequency,
            power_factor,
            temperature,
            thd
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probabilities = model.predict_proba(input_scaled)[0]

        confidence = float(np.max(probabilities) * 100)

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence

        st.rerun()


    # ========================================================
    # RESULT
    # ========================================================

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction
        confidence = st.session_state.confidence

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == "Normal":

            result_class = "result result-normal"

            description = (
                "The supplied electrical measurements are "
                "classified as operating within the normal condition."
            )

        else:

            result_class = "result result-fault"

            description = (
                "The machine-learning model has identified "
                "an abnormal electrical operating condition."
            )


        st.markdown(
            f"""
            <div class="{result_class}">

                <div class="result-label">
                    AI CLASSIFICATION RESULT
                </div>

                <div class="result-value">
                    {prediction}
                </div>

                <div class="result-confidence">
                    {description}
                </div>

                <br>

                <div class="result-confidence">
                    Model confidence: <strong>{confidence:.2f}%</strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Input measurements")

        measurements = pd.DataFrame({
            "Parameter": [
                "Voltage",
                "Current",
                "Frequency",
                "Power Factor",
                "Temperature",
                "THD"
            ],
            "Value": [
                f"{voltage:.2f} V",
                f"{current:.2f} A",
                f"{frequency:.2f} Hz",
                f"{power_factor:.2f}",
                f"{temperature:.2f} °C",
                f"{thd:.2f}%"
            ]
        })

        st.dataframe(
            measurements,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SYSTEM PAGE
# ============================================================

elif st.session_state.page == "System":

    st.markdown("""
    <div class="dashboard">

        <div class="dashboard-title">
            System Architecture
        </div>

        <div class="dashboard-description">
            Overview of the machine-learning pipeline used by
            ElectroGuard AI.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="fault">

            <div class="fault-code">
                INPUT LAYER
            </div>

            <div class="fault-title">
                Electrical parameters
            </div>

            <div class="fault-description">
                Voltage, current, frequency, power factor,
                temperature and THD form the input feature vector.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="fault">

            <div class="fault-code">
                PROCESSING
            </div>

            <div class="fault-title">
                StandardScaler
            </div>

            <div class="fault-description">
                Input values are transformed using the same
                scaling process used during model training.
            </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("""
        <div class="fault">

            <div class="fault-code">
                ML ENGINE
            </div>

            <div class="fault-title">
                Random Forest Classifier
            </div>

            <div class="fault-description">
                An ensemble of 100 decision trees evaluates
                the electrical operating condition.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col4:

        st.markdown("""
        <div class="fault">

            <div class="fault-code">
                OUTPUT
            </div>

            <div class="fault-title">
                Fault classification
            </div>

            <div class="fault-description">
                The model returns one of four conditions:
                Normal, Overvoltage, Undervoltage or Overcurrent.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    st.markdown("""
    <div class="dashboard">

        <div class="dashboard-title">
            About ElectroGuard AI
        </div>

        <div class="dashboard-description">
            An AI-based prototype for intelligent electrical
            fault classification.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="fault">

        <div class="fault-code">
            PROJECT OBJECTIVE
        </div>

        <div class="fault-title">
            Early identification of abnormal electrical conditions
        </div>

        <div class="fault-description">
            ElectroGuard AI analyzes multiple electrical parameters
            simultaneously and uses machine learning to classify
            the current operating condition of a power system.

            The current prototype is trained using synthetic data
            and demonstrates the complete workflow from data
            preprocessing to model prediction and web deployment.

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)

    tech = [
        ("PYTHON", "Core development"),
        ("SCIKIT-LEARN", "ML framework"),
        ("RANDOM FOREST", "Classification model"),
        ("STREAMLIT", "Web application")
    ]

    for col, (title, desc) in zip(cols, tech):

        with col:

            st.markdown(
                f"""
                <div class="stat">

                    <div class="stat-number"
                         style="font-size:18px;">
                        {title}
                    </div>

                    <div class="stat-label">
                        {desc}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    ELECTROGUARD AI · AI-BASED ELECTRICAL FAULT DETECTION · ML PROTOTYPE
</div>
""", unsafe_allow_html=True)
