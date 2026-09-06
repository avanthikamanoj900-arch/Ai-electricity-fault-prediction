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
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0


# =========================================================
# CSS
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
    background: #070d17;
    color: #ffffff;
}

.block-container {
    max-width: 1450px;
    padding: 25px 55px 60px 55px;
}

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ================= NAVBAR ================= */

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 65px;
    border-bottom: 1px solid #172333;
    margin-bottom: 35px;
}

.logo-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: #1683ff;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 21px;
}

.brand {
    font-size: 20px;
    font-weight: 800;
}

.brand-small {
    font-size: 9px;
    color: #61748a;
    letter-spacing: 1.5px;
}

.online {
    color: #30d98a;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #30d98a;
    box-shadow: 0 0 10px #30d98a;
}


/* ================= HERO ================= */

.hero {
    min-height: 490px;
    border-radius: 24px;
    padding: 65px;
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(
            circle at 80% 40%,
            rgba(19,133,255,0.18),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #0a1a2d,
            #071321
        );
    border: 1px solid #17283b;
}

.grid {
    position: absolute;
    inset: 0;
    opacity: 0.13;
    background-image:
        linear-gradient(#3b76a5 1px, transparent 1px),
        linear-gradient(90deg, #3b76a5 1px, transparent 1px);
    background-size: 50px 50px;
}

.hero-content {
    position: relative;
    z-index: 2;
    width: 58%;
}

.tag {
    display: inline-block;
    padding: 8px 14px;
    border: 1px solid #1b6dab;
    border-radius: 30px;
    color: #42aaff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    background: rgba(20,130,255,0.07);
}

.hero-title {
    font-size: 58px;
    line-height: 1.04;
    font-weight: 800;
    letter-spacing: -2px;
    margin-top: 22px;
}

.blue {
    color: #36a5ff;
}

.hero-text {
    color: #8498ac;
    font-size: 16px;
    line-height: 1.8;
    max-width: 610px;
    margin-top: 20px;
}


/* ================= MONITOR ================= */

.monitor {
    position: absolute;
    right: 55px;
    top: 65px;
    width: 390px;
    height: 355px;
    background: #081522;
    border: 1px solid #19354d;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.4);
}

.monitor-header {
    display: flex;
    justify-content: space-between;
    color: #8195a9;
    font-size: 11px;
}

.live {
    color: #28d58a;
}

.wave {
    margin-top: 25px;
    height: 155px;
    background:
        linear-gradient(#102c40 1px, transparent 1px),
        linear-gradient(90deg, #102c40 1px, transparent 1px);
    background-size: 30px 30px;
    border-radius: 10px;
}

.wave svg {
    width: 100%;
    height: 100%;
}

.monitor-data {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 10px;
    margin-top: 18px;
}

.data {
    background: #0d1c2b;
    border-radius: 9px;
    padding: 12px;
}

.data-label {
    color: #5f7489;
    font-size: 9px;
}

.data-value {
    font-size: 17px;
    font-weight: 700;
    margin-top: 4px;
}


/* ================= SECTIONS ================= */

.section {
    margin-top: 75px;
}

.kicker {
    color: #2799ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
}

.title {
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.subtitle {
    color: #71869b;
    font-size: 14px;
    max-width: 700px;
    line-height: 1.7;
    margin-top: 8px;
}


/* ================= STATS ================= */

.stat-card {
    background: #0b1624;
    border: 1px solid #172638;
    border-radius: 14px;
    padding: 23px;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}

.stat-label {
    color: #63788d;
    font-size: 11px;
    margin-top: 5px;
}


/* ================= PROCESS ================= */

.process {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    margin-top: 35px;
}

.process-card {
    padding: 25px;
    border-left: 1px solid #1a2b3d;
    min-height: 170px;
}

.process-card:first-child {
    border-left: 2px solid #208fff;
}

.number {
    color: #208fff;
    font-size: 11px;
    font-weight: 800;
}

.process-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 15px;
}

.process-text {
    color: #657a8e;
    font-size: 12px;
    line-height: 1.65;
    margin-top: 8px;
}


/* ================= FAULT CARDS ================= */

.fault-card {
    background: #0b1624;
    border: 1px solid #172638;
    border-radius: 14px;
    padding: 25px;
    min-height: 145px;
}

.fault-code {
    color: #278fff;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.fault-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 13px;
}

.fault-text {
    color: #667c90;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 7px;
}


/* ================= DASHBOARD ================= */

.dashboard-header {
    padding: 30px;
    background: linear-gradient(
        135deg,
        #0d263e,
        #0a1a2c
    );
    border-radius: 18px;
    border: 1px solid #183249;
}

.dashboard-title {
    font-size: 30px;
    font-weight: 800;
}

.dashboard-text {
    color: #7890a5;
    font-size: 13px;
    margin-top: 7px;
}


/* ================= RESULT ================= */

.result-normal {
    padding: 30px;
    border-radius: 15px;
    background: rgba(31,210,135,0.07);
    border: 1px solid rgba(31,210,135,0.25);
}

.result-fault {
    padding: 30px;
    border-radius: 15px;
    background: rgba(255,67,67,0.07);
    border: 1px solid rgba(255,67,67,0.25);
}

.result-label {
    font-size: 9px;
    letter-spacing: 2px;
    color: #71879a;
}

.result-value {
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.result-info {
    color: #8296a9;
    font-size: 13px;
    margin-top: 8px;
}


/* ================= INPUTS ================= */

label {
    color: #a7b8c8 !important;
}

div[data-baseweb="input"] {
    background: #0c1927 !important;
}

input {
    color: white !important;
}


/* ================= FOOTER ================= */

.footer {
    margin-top: 90px;
    padding-top: 25px;
    border-top: 1px solid #172333;
    text-align: center;
    color: #506478;
    font-size: 10px;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NAVBAR
# =========================================================

st.markdown("""
<div class="navbar">

    <div class="logo-wrapper">

        <div class="logo">⚡</div>

        <div>
            <div class="brand">ElectroGuard AI</div>
            <div class="brand-small">
                INTELLIGENT POWER MONITORING
            </div>
        </div>

    </div>

    <div class="online">
        <div class="online-dot"></div>
        SYSTEM ONLINE
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

n1, n2, n3, n4 = st.columns(4)

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


# =========================================================
# OVERVIEW
# =========================================================

if st.session_state.page == "Overview":

    st.markdown("""
    <div class="hero">

        <div class="grid"></div>

        <div class="hero-content">

            <div class="tag">
                AI-BASED ELECTRICAL FAULT DETECTION
            </div>

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
                <span class="live">● LIVE</span>
            </div>

            <div class="wave">

                <svg viewBox="0 0 360 155"
                     preserveAspectRatio="none">

                    <polyline
                        points="
                        0,78
                        15,78
                        25,35
                        40,120
                        55,78
                        75,78
                        90,42
                        105,110
                        120,78
                        145,78
                        160,30
                        175,125
                        190,78
                        215,78
                        230,40
                        245,115
                        260,78
                        285,78
                        300,32
                        315,122
                        330,78
                        360,78"
                        fill="none"
                        stroke="#229cff"
                        stroke-width="3"
                    />

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

    c1, c2, c3 = st.columns([1,1,2])

    with c1:
        if st.button(
            "START ANALYSIS →",
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


    # =====================================================
    # MODEL
    # =====================================================

    st.markdown("""
    <div class="section">

        <div class="kicker">
            MODEL PERFORMANCE
        </div>

        <div class="title">
            Machine-learning system overview
        </div>

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


    # =====================================================
    # PIPELINE
    # =====================================================

    st.markdown("""
    <div class="section">

        <div class="kicker">
            HOW IT WORKS
        </div>

        <div class="title">
            From electrical signal to AI decision
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="process">

        <div class="process-card">
            <div class="number">01</div>
            <div class="process-title">Measure</div>
            <div class="process-text">
                Electrical operating parameters are collected
                from the monitored system.
            </div>
        </div>

        <div class="process-card">
            <div class="number">02</div>
            <div class="process-title">Normalize</div>
            <div class="process-text">
                Measurements are transformed using the same
                preprocessing pipeline used during training.
            </div>
        </div>

        <div class="process-card">
            <div class="number">03</div>
            <div class="process-title">Classify</div>
            <div class="process-text">
                The Random Forest model evaluates the feature
                pattern and determines the operating condition.
            </div>
        </div>

        <div class="process-card">
            <div class="number">04</div>
            <div class="process-title">Report</div>
            <div class="process-text">
                The detected condition and model confidence
                are presented to the operator.
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # FAULTS
    # =====================================================

    st.markdown("""
    <div class="section">

        <div class="kicker">
            FAULT CLASSIFICATION
        </div>

        <div class="title">
            What ElectroGuard can detect
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)

    faults = [
        (
            "F-01",
            "Normal",
            "Operating parameters remain within the expected range."
        ),
        (
            "F-02",
            "Overvoltage",
            "Voltage rises above the expected operating condition."
        ),
        (
            "F-03",
            "Undervoltage",
            "Voltage falls below the expected operating condition."
        ),
        (
            "F-04",
            "Overcurrent",
            "Current exceeds the expected operating condition."
        )
    ]

    for col, (code, title, text) in zip(cols, faults):

        with col:

            st.markdown(
                f"""
                <div class="fault-card">

                    <div class="fault-code">{code}</div>

                    <div class="fault-title">
                        {title}
                    </div>

                    <div class="fault-text">
                        {text}
                    </div>

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

        <div class="dashboard-title">
            Electrical Fault Analysis
        </div>

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

            <div class="fault-code">
                INPUT PARAMETERS
            </div>

            <div class="fault-title">
                Electrical measurements
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:

            voltage = st.number_input(
                "Voltage (V)",
                0.0,
                500.0,
                230.0,
                1.0
            )

            current = st.number_input(
                "Current (A)",
                0.0,
                500.0,
                5.0,
                0.5
            )

            frequency = st.number_input(
                "Frequency (Hz)",
                0.0,
                100.0,
                50.0,
                0.1
            )

        with c2:

            power_factor = st.number_input(
                "Power Factor",
                0.0,
                1.0,
                0.95,
                0.01
            )

            temperature = st.number_input(
                "Temperature (°C)",
                -20.0,
                150.0,
                30.0,
                1.0
            )

            thd = st.number_input(
                "THD (%)",
                0.0,
                100.0,
                2.0,
                0.5
            )


        if st.button(
            "RUN AI FAULT ANALYSIS",
            type="primary",
            use_container_width=True
        ):

            data = np.array([[
                voltage,
                current,
                frequency,
                power_factor,
                temperature,
                thd
            ]])

            scaled_data = scaler.transform(data)

            prediction = model.predict(scaled_data)[0]

            probabilities = model.predict_proba(scaled_data)[0]

            confidence = float(
                np.max(probabilities) * 100
            )

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence

            st.rerun()


    with col2:

        st.markdown("""
        <div class="fault-card">

            <div class="fault-code">
                AI OUTPUT
            </div>

            <div class="fault-title">
                Prediction result
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.prediction is None:

            st.info(
                "Enter the electrical parameters and run "
                "the AI analysis."
            )

        else:

            prediction = st.session_state.prediction
            confidence = st.session_state.confidence

            if prediction == "Normal":

                st.markdown(
                    f"""
                    <div class="result-normal">

                        <div class="result-label">
                            OPERATING CONDITION
                        </div>

                        <div class="result-value">
                            NORMAL
                        </div>

                        <div class="result-info">
                            No abnormal electrical condition
                            was identified.
                        </div>

                        <br>

                        <div class="result-info">
                            Model confidence:
                            <strong>{confidence:.2f}%</strong>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-fault">

                        <div class="result-label">
                            FAULT DETECTED
                        </div>

                        <div class="result-value">
                            {prediction.upper()}
                        </div>

                        <div class="result-info">
                            The model identified an abnormal
                            electrical operating condition.
                        </div>

                        <br>

                        <div class="result-info">
                            Model confidence:
                            <strong>{confidence:.2f}%</strong>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # =====================================================
    # MEASUREMENTS
    # =====================================================

    if st.session_state.prediction is not None:

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(
            "### Measurement Summary"
        )

        table = pd.DataFrame({
            "Parameter": [
                "Voltage",
                "Current",
                "Frequency",
                "Power Factor",
                "Temperature",
                "THD"
            ],
            "Measured Value": [
                f"{voltage:.2f} V",
                f"{current:.2f} A",
                f"{frequency:.2f} Hz",
                f"{power_factor:.2f}",
                f"{temperature:.2f} °C",
                f"{thd:.2f}%"
            ]
        })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# SYSTEM ARCHITECTURE
# =========================================================

elif st.session_state.page == "System":

    st.markdown("""
    <div class="dashboard-header">

        <div class="dashboard-title">
            System Architecture
        </div>

        <div class="dashboard-text">
            The complete machine-learning pipeline used by
            ElectroGuard AI.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)

    architecture = [
        (
            "01",
            "Electrical Inputs",
            "Voltage, current, frequency, power factor, temperature and THD."
        ),
        (
            "02",
            "Preprocessing",
            "StandardScaler transforms the incoming feature values."
        ),
        (
            "03",
            "ML Model",
            "Random Forest evaluates the processed feature vector."
        ),
        (
            "04",
            "Fault Output",
            "Normal, Overvoltage, Undervoltage or Overcurrent."
        )
    ]

    for col, (number, title, text) in zip(cols, architecture):

        with col:

            st.markdown(
                f"""
                <div class="fault-card">

                    <div class="fault-code">
                        STAGE {number}
                    </div>

                    <div class="fault-title">
                        {title}
                    </div>

                    <div class="fault-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About":

    st.markdown("""
    <div class="dashboard-header">

        <div class="dashboard-title">
            About ElectroGuard AI
        </div>

        <div class="dashboard-text">
            AI-based electrical fault detection and classification.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="fault-card">

        <div class="fault-code">
            PROJECT
        </div>

        <div class="fault-title">
            Intelligent electrical monitoring
        </div>

        <div class="fault-text">

            ElectroGuard AI is a machine-learning prototype designed
            to analyze electrical operating parameters and classify
            potential faults.

            The system uses six electrical features and a Random Forest
            classifier to distinguish between normal operation,
            overvoltage, undervoltage and overcurrent conditions.

            The current model is trained using synthetic prototype data.
            Real-world deployment would require validation using
            measured electrical data from actual power systems.

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)

    technologies = [
        ("PYTHON", "Core development"),
        ("SCIKIT-LEARN", "Machine learning"),
        ("RANDOM FOREST", "Fault classification"),
        ("STREAMLIT", "Web application")
    ]

    for col, (title, text) in zip(cols, technologies):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-number"
                         style="font-size:18px;">
                        {title}
                    </div>

                    <div class="stat-label">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    ELECTROGUARD AI  •  AI-BASED ELECTRICAL FAULT DETECTION
</div>
""", unsafe_allow_html=True)
