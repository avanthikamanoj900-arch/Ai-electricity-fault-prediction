import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ElectroGuard AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("electrical_fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f9fc;
    color: #172033;
}

/* Remove Streamlit default spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Hide default Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Navigation */
.navbar {
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 5px;
    margin-bottom: 35px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 42px;
    height: 42px;
    border-radius: 11px;
    background: #123c69;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 800;
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: #14213d;
}

.brand-sub {
    font-size: 11px;
    color: #7a8496;
    margin-top: -2px;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f2947 0%, #164e78 100%);
    border-radius: 26px;
    padding: 65px 65px;
    min-height: 455px;
    position: relative;
    overflow: hidden;
    color: white;
}

.hero::after {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 50%;
    right: -120px;
    top: -130px;
}

.hero::before {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50%;
    right: 70px;
    bottom: -190px;
}

.hero-content {
    max-width: 680px;
    position: relative;
    z-index: 2;
}

.hero-label {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 12px;
    letter-spacing: 0.4px;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 52px;
    line-height: 1.08;
    margin: 0;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero p {
    font-size: 17px;
    line-height: 1.7;
    color: #d8e6f2;
    max-width: 600px;
    margin-top: 23px;
}

.hero-stat {
    position: absolute;
    right: 65px;
    top: 115px;
    width: 275px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 24px;
    z-index: 2;
    backdrop-filter: blur(8px);
}

.hero-stat-title {
    font-size: 12px;
    color: #bcd0e2;
    margin-bottom: 15px;
}

.hero-reading {
    font-size: 34px;
    font-weight: 800;
}

.hero-reading-label {
    color: #c5d8e8;
    font-size: 12px;
}

.wave {
    height: 70px;
    margin-top: 22px;
    position: relative;
    overflow: hidden;
}

.wave svg {
    width: 100%;
    height: 100%;
}

/* Section */
.section {
    margin-top: 75px;
}

.section-label {
    color: #2878b8;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}

.section-title {
    font-size: 34px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 12px;
}

.section-description {
    color: #667085;
    font-size: 15px;
    line-height: 1.7;
    max-width: 700px;
}

/* Stats */
.stat-box {
    background: white;
    border: 1px solid #e5eaf0;
    border-radius: 16px;
    padding: 25px;
    height: 100%;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #123c69;
}

.stat-label {
    color: #6b7280;
    font-size: 13px;
    margin-top: 5px;
}

/* Process */
.process-row {
    display: flex;
    gap: 0;
    margin-top: 35px;
}

.process-item {
    flex: 1;
    padding: 28px;
    border-top: 2px solid #d9e1e9;
    position: relative;
}

.process-item:first-child {
    border-top-color: #2878b8;
}

.process-number {
    font-size: 12px;
    color: #2878b8;
    font-weight: 800;
    margin-bottom: 16px;
}

.process-title {
    font-size: 18px;
    font-weight: 700;
    color: #172033;
    margin-bottom: 9px;
}

.process-text {
    font-size: 13px;
    line-height: 1.6;
    color: #6b7280;
}

/* Fault section */
.fault-box {
    background: white;
    border: 1px solid #e4e9ef;
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
}

.fault-name {
    font-size: 18px;
    font-weight: 700;
    color: #172033;
}

.fault-desc {
    color: #6b7280;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
}

/* Login */
.login-container {
    max-width: 470px;
    margin: 80px auto;
}

.login-title {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    color: #172033;
}

.login-subtitle {
    text-align: center;
    color: #667085;
    margin-bottom: 35px;
}

/* Dashboard */
.dashboard-header {
    background: #123c69;
    border-radius: 20px;
    padding: 35px;
    color: white;
    margin-bottom: 30px;
}

.dashboard-title {
    font-size: 32px;
    font-weight: 800;
}

.dashboard-sub {
    color: #c7d9e8;
    margin-top: 8px;
}

/* Prediction */
.prediction-panel {
    background: white;
    border: 1px solid #e3e8ee;
    border-radius: 18px;
    padding: 30px;
}

.result-normal {
    background: #edf8f1;
    border-left: 5px solid #2e8b57;
    padding: 25px;
    border-radius: 10px;
}

.result-fault {
    background: #fff3f1;
    border-left: 5px solid #d64545;
    padding: 25px;
    border-radius: 10px;
}

.result-title {
    font-size: 24px;
    font-weight: 800;
}

.result-sub {
    color: #667085;
    margin-top: 6px;
}

/* Buttons */
.stButton > button {
    border-radius: 9px;
    height: 43px;
    font-weight: 600;
    border: 1px solid #d6dde6;
}

.stButton > button:hover {
    border-color: #2878b8;
}

/* Inputs */
div[data-baseweb="input"] {
    border-radius: 8px;
}

/* Footer */
.footer {
    margin-top: 90px;
    padding-top: 25px;
    border-top: 1px solid #e2e7ed;
    text-align: center;
    color: #8a94a3;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

st.markdown("""
<div class="navbar">

    <div class="brand">
        <div class="brand-logo">EG</div>
        <div>
            <div class="brand-name">ElectroGuard AI</div>
            <div class="brand-sub">INTELLIGENT POWER MONITORING</div>
        </div>
    </div>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("### ElectroGuard AI")

    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"

    if st.button("Fault Detection", use_container_width=True):
        st.session_state.page = "Detection"

    if st.button("About System", use_container_width=True):
        st.session_state.page = "About"

    st.markdown("---")

    if not st.session_state.logged_in:

        if st.button("Login", use_container_width=True):
            st.session_state.page = "Login"

    else:

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.rerun()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-content">

            <div class="hero-label">
                AI-BASED ELECTRICAL FAULT DETECTION
            </div>

            <h1>
                Detect electrical faults
                before they escalate.
            </h1>

            <p>
                ElectroGuard AI analyzes electrical operating conditions
                using machine learning to identify abnormal behavior
                and classify potential faults in real time.
            </p>

        </div>

        <div class="hero-stat">

            <div class="hero-stat-title">
                LIVE SYSTEM REFERENCE
            </div>

            <div class="hero-reading">
                230.4 V
            </div>

            <div class="hero-reading-label">
                Nominal operating voltage
            </div>

            <div class="wave">
                <svg viewBox="0 0 300 70">
                    <polyline
                        points="0,35 20,35 35,15 50,55 65,35
                        90,35 105,10 120,60 135,35
                        160,35 175,18 190,52 205,35
                        230,35 245,12 260,58 275,35 300,35"
                        fill="none"
                        stroke="#73b8e6"
                        stroke-width="2.5"/>
                </svg>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    st.markdown("""
    <div class="section">

        <div class="section-label">SYSTEM OVERVIEW</div>

        <div class="section-title">
            From electrical measurements to an intelligent decision
        </div>

        <div class="section-description">
            The system combines electrical measurements with a trained
            Random Forest classification model to identify the operating
            condition of a power system.
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)

    stats = [
        ("91.67%", "Prototype model accuracy"),
        ("4", "Fault conditions"),
        ("6", "Electrical input parameters"),
        ("100", "Random Forest estimators")
    ]

    for col, (number, label) in zip(cols, stats):

        with col:

            st.markdown(
                f"""
                <div class="stat-box">

                    <div class="stat-number">{number}</div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------

    st.markdown("""
    <div class="section">

        <div class="section-label">HOW IT WORKS</div>

        <div class="section-title">
            A four-stage detection pipeline
        </div>

        <div class="section-description">
            Every prediction follows the same processing sequence used
            during model development.
        </div>

    </div>
    """, unsafe_allow_html=True)


    processes = [
        (
            "01",
            "Measure",
            "Voltage, current, frequency, power factor, temperature and THD are collected."
        ),
        (
            "02",
            "Pre-process",
            "The incoming measurements are scaled using the trained preprocessing pipeline."
        ),
        (
            "03",
            "Classify",
            "The Random Forest model evaluates the electrical condition and identifies the fault class."
        ),
        (
            "04",
            "Report",
            "The predicted condition and confidence are presented to the operator."
        )
    ]

    process_html = '<div class="process-row">'

    for number, title, text in processes:

        process_html += f"""
        <div class="process-item">

            <div class="process-number">{number}</div>

            <div class="process-title">{title}</div>

            <div class="process-text">{text}</div>

        </div>
        """

    process_html += "</div>"

    st.markdown(process_html, unsafe_allow_html=True)


    # --------------------------------------------------
    # DETECTION CAPABILITIES
    # --------------------------------------------------

    st.markdown("""
    <div class="section">

        <div class="section-label">DETECTION CAPABILITIES</div>

        <div class="section-title">
            Electrical conditions the model can identify
        </div>

    </div>
    """, unsafe_allow_html=True)


    faults = [
        (
            "Normal",
            "Electrical parameters remain within the expected operating range."
        ),
        (
            "Overvoltage",
            "Voltage rises above the expected operating condition."
        ),
        (
            "Undervoltage",
            "Voltage falls below the expected operating condition."
        ),
        (
            "Overcurrent",
            "Current rises beyond the expected operating condition."
        )
    ]

    for name, description in faults:

        st.markdown(
            f"""
            <div class="fault-box">

                <div class="fault-name">
                    {name}
                </div>

                <div class="fault-desc">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

elif st.session_state.page == "Login":

    st.markdown(
        """
        <div class="login-container">

            <div class="login-title">
                Welcome back
            </div>

            <div class="login-subtitle">
                Sign in to access the ElectroGuard monitoring system.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button(
            "Sign in",
            use_container_width=True,
            type="primary"
        ):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.page = "Detection"

                st.success("Login successful.")
                st.rerun()

            else:

                st.error("Invalid username or password.")

        st.caption("Demo credentials: admin / admin123")


# --------------------------------------------------
# DETECTION PAGE
# --------------------------------------------------

elif st.session_state.page == "Detection":

    if not st.session_state.logged_in:

        st.warning("Please login to access the fault detection system.")

        if st.button("Go to Login"):
            st.session_state.page = "Login"
            st.rerun()

    else:

        st.markdown(
            """
            <div class="dashboard-header">

                <div class="dashboard-title">
                    Electrical Fault Detection
                </div>

                <div class="dashboard-sub">
                    Enter electrical operating parameters to evaluate
                    the current system condition.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # --------------------------------------------------
        # INPUT PANEL
        # --------------------------------------------------

        st.markdown(
            """
            <div class="prediction-panel">
                <h3>Electrical measurements</h3>
                <p style="color:#667085;">
                    Provide the measured electrical parameters below.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

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
            "Run Fault Analysis",
            use_container_width=True,
            type="primary"
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

            confidence = np.max(probabilities) * 100


            st.markdown("<br>", unsafe_allow_html=True)

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            if prediction == "Normal":

                st.markdown(
                    f"""
                    <div class="result-normal">

                        <div class="result-title">
                            System condition: Normal
                        </div>

                        <div class="result-sub">
                            No abnormal electrical condition was detected
                            for the supplied measurements.
                        </div>

                        <br>

                        <strong>Model confidence: {confidence:.2f}%</strong>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-fault">

                        <div class="result-title">
                            Fault detected: {prediction}
                        </div>

                        <div class="result-sub">
                            The machine learning model identified an
                            abnormal electrical operating condition.
                        </div>

                        <br>

                        <strong>Model confidence: {confidence:.2f}%</strong>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # --------------------------------------------------
            # MEASUREMENT SUMMARY
            # --------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Measurement summary")

            result_df = pd.DataFrame({
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
                result_df,
                use_container_width=True,
                hide_index=True
            )


            # --------------------------------------------------
            # MODEL PROBABILITIES
            # --------------------------------------------------

            st.subheader("Model probability")

            probability_df = pd.DataFrame({
                "Fault Class": model.classes_,
                "Probability": probabilities
            })

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.bar_chart(
                probability_df.set_index("Fault Class")
            )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif st.session_state.page == "About":

    st.markdown("""
    <div class="section">

        <div class="section-label">ABOUT THE PROJECT</div>

        <div class="section-title">
            ElectroGuard AI
        </div>

        <div class="section-description">

            ElectroGuard AI is a machine-learning based electrical
            fault detection prototype designed to analyze electrical
            operating parameters and classify system conditions.

            The system uses six input parameters:

            voltage, current, frequency, power factor, temperature
            and total harmonic distortion.

        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="fault-box">

            <div class="fault-name">
                Machine Learning Model
            </div>

            <div class="fault-desc">
                A Random Forest classifier is trained to distinguish
                between Normal, Overvoltage, Undervoltage and
                Overcurrent operating conditions.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="fault-box">

            <div class="fault-name">
                Prototype Dataset
            </div>

            <div class="fault-desc">
                The current dataset is synthetic prototype data created
                for demonstrating the machine-learning workflow.
                Real-world validation would require measured electrical
                data from actual systems.
            </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <div class="section">

        <div class="section-label">TECHNOLOGY</div>

        <div class="section-title">
            Built with a practical ML pipeline
        </div>

    </div>
    """, unsafe_allow_html=True)


    tech = st.columns(4)

    technologies = [
        ("Python", "Core development"),
        ("Scikit-learn", "Machine learning"),
        ("Random Forest", "Fault classification"),
        ("Streamlit", "Web application")
    ]

    for col, (name, description) in zip(tech, technologies):

        with col:

            st.markdown(
                f"""
                <div class="stat-box">

                    <div class="stat-number"
                         style="font-size:20px;">
                        {name}
                    </div>

                    <div class="stat-label">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

    ElectroGuard AI · AI-Based Electrical Fault Detection System

</div>
""", unsafe_allow_html=True)
