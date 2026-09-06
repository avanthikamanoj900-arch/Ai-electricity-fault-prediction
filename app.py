import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ElectroGuard AI",
    page_icon="",
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
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background: #f6f8fb;
    color: #172033;
}

header {
    visibility: hidden;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

* {
    font-family: "Inter", "Segoe UI", sans-serif;
}


/* ---------- NAVIGATION ---------- */

.top-nav {
    height: 65px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #e3e8ef;
    margin-bottom: 20px;
}

.brand {
    font-size: 21px;
    font-weight: 750;
    color: #12233f;
    letter-spacing: -0.4px;
}

.brand-mark {
    color: #1769aa;
}

.nav-caption {
    font-size: 12px;
    color: #718096;
    letter-spacing: 1px;
}


/* ---------- HERO ---------- */

.hero {
    padding: 80px 0 75px 0;
}

.eyebrow {
    font-size: 12px;
    font-weight: 700;
    color: #1769aa;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 62px;
    line-height: 1.04;
    letter-spacing: -2.5px;
    font-weight: 780;
    color: #10233f;
    margin-bottom: 24px;
}

.hero-title span {
    color: #1769aa;
}

.hero-text {
    max-width: 650px;
    font-size: 18px;
    line-height: 1.75;
    color: #64748b;
    margin-bottom: 30px;
}


/* ---------- SYSTEM PANEL ---------- */

.system-panel {
    background: #ffffff;
    border: 1px solid #dce3eb;
    border-radius: 14px;
    padding: 30px;
    box-shadow: 0 12px 35px rgba(16, 35, 63, 0.07);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #e8edf2;
    padding-bottom: 18px;
    margin-bottom: 24px;
}

.panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #24364f;
    letter-spacing: 0.8px;
}

.online {
    font-size: 12px;
    font-weight: 700;
    color: #23815b;
}

.large-reading {
    font-size: 48px;
    font-weight: 750;
    color: #12233f;
    line-height: 1;
}

.reading-unit {
    color: #718096;
    font-size: 14px;
}

.parameter-row {
    display: flex;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid #edf1f5;
}

.parameter-name {
    color: #64748b;
    font-size: 13px;
}

.parameter-value {
    color: #172033;
    font-weight: 700;
    font-size: 14px;
}


/* ---------- SECTION ---------- */

.section {
    padding: 45px 0;
}

.section-label {
    color: #1769aa;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.section-title {
    color: #12233f;
    font-size: 34px;
    font-weight: 750;
    letter-spacing: -1px;
    margin-top: 8px;
}

.section-description {
    color: #718096;
    max-width: 650px;
    line-height: 1.7;
}


/* ---------- STATS ---------- */

.stat {
    background: #ffffff;
    border-top: 3px solid #1769aa;
    padding: 25px;
    border-radius: 4px;
    border-left: 1px solid #e1e7ee;
    border-right: 1px solid #e1e7ee;
    border-bottom: 1px solid #e1e7ee;
}

.stat-value {
    font-size: 31px;
    font-weight: 750;
    color: #12233f;
}

.stat-label {
    font-size: 11px;
    color: #718096;
    letter-spacing: 0.8px;
    margin-top: 5px;
}


/* ---------- HOW IT WORKS ---------- */

.process {
    position: relative;
    background: #ffffff;
    border: 1px solid #dfe5ec;
    border-radius: 12px;
    padding: 30px 25px;
    min-height: 205px;
}

.process-number {
    color: #1769aa;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1px;
}

.process-title {
    color: #172033;
    font-size: 20px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 12px;
}

.process-description {
    color: #718096;
    font-size: 13px;
    line-height: 1.7;
}


/* ---------- FEATURE SECTION ---------- */

.feature {
    border-bottom: 1px solid #dfe5ec;
    padding: 25px 0;
}

.feature-title {
    color: #172033;
    font-size: 17px;
    font-weight: 700;
}

.feature-text {
    color: #718096;
    font-size: 13px;
    line-height: 1.65;
}


/* ---------- LOGIN ---------- */

.login-box {
    background: #ffffff;
    border: 1px solid #dfe5ec;
    border-radius: 12px;
    padding: 45px;
    box-shadow: 0 15px 40px rgba(16,35,63,0.08);
}

.login-title {
    color: #12233f;
    font-size: 30px;
    font-weight: 750;
    text-align: center;
}

.login-subtitle {
    color: #718096;
    text-align: center;
    font-size: 13px;
    margin-bottom: 30px;
}


/* ---------- DASHBOARD ---------- */

.dashboard-header {
    margin-bottom: 30px;
}

.dashboard-title {
    font-size: 35px;
    font-weight: 750;
    color: #12233f;
}

.dashboard-subtitle {
    color: #718096;
    font-size: 14px;
}


/* ---------- MONITORING CARDS ---------- */

.monitor-card {
    background: #ffffff;
    border: 1px solid #dfe5ec;
    border-radius: 10px;
    padding: 25px;
}

.monitor-label {
    color: #718096;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.monitor-value {
    color: #12233f;
    font-size: 30px;
    font-weight: 750;
    margin-top: 8px;
}


/* ---------- RESULT ---------- */

.normal-result {
    background: #f1faf6;
    border: 1px solid #b9e2cf;
    border-left: 5px solid #23815b;
    padding: 30px;
    border-radius: 8px;
}

.normal-title {
    color: #1f7655;
    font-size: 25px;
    font-weight: 750;
}

.fault-result {
    background: #fff5f4;
    border: 1px solid #edc1bd;
    border-left: 5px solid #c7463c;
    padding: 30px;
    border-radius: 8px;
}

.fault-title {
    color: #b33e36;
    font-size: 25px;
    font-weight: 750;
}


/* ---------- BUTTONS ---------- */

.stButton > button {
    background: #1769aa;
    color: white;
    border: none;
    border-radius: 6px;
    min-height: 45px;
    font-weight: 650;
    padding: 0 24px;
}

.stButton > button:hover {
    background: #12598f;
    color: white;
}


/* ---------- INPUTS ---------- */

.stTextInput input,
.stNumberInput input {
    background: #ffffff;
    border: 1px solid #ccd5df;
    border-radius: 6px;
    color: #172033;
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #12233f;
}

section[data-testid="stSidebar"] * {
    color: white;
}


/* ---------- TABLE ---------- */

.dataframe {
    border: 1px solid #dfe5ec;
}


/* ---------- FOOTER ---------- */

.footer {
    border-top: 1px solid #dfe5ec;
    margin-top: 70px;
    padding-top: 25px;
    color: #8a97a8;
    font-size: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LANDING PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown("""
    <div class="top-nav">

        <div class="brand">
            <span class="brand-mark">EG</span> ElectroGuard AI
        </div>

        <div class="nav-caption">
            ELECTRICAL INTELLIGENCE PLATFORM
        </div>

    </div>
    """, unsafe_allow_html=True)


    # ---------------- HERO ----------------

    left, right = st.columns([1.15, 0.85])

    with left:

        st.markdown("""
        <div class="hero">

            <div class="eyebrow">
                Machine Learning for Electrical Systems
            </div>

            <div class="hero-title">
                Detect electrical faults<br>
                <span>before they escalate.</span>
            </div>

            <div class="hero-text">
                ElectroGuard AI analyzes voltage, current, frequency,
                power factor, temperature and harmonic distortion
                to classify abnormal electrical operating conditions.
            </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Start system analysis",
            use_container_width=False
        ):
            st.session_state.page = "login"
            st.rerun()


    with right:

        st.markdown("""
        <div class="system-panel">

            <div class="panel-header">

                <div class="panel-title">
                    SYSTEM OVERVIEW
                </div>

                <div class="online">
                    ● ONLINE
                </div>

            </div>

            <div style="color:#718096;font-size:12px;">
                SUPPLY VOLTAGE
            </div>

            <div class="large-reading">
                230 <span class="reading-unit">V</span>
            </div>

            <br>

            <div class="parameter-row">
                <div class="parameter-name">Current</div>
                <div class="parameter-value">5.0 A</div>
            </div>

            <div class="parameter-row">
                <div class="parameter-name">Frequency</div>
                <div class="parameter-value">50 Hz</div>
            </div>

            <div class="parameter-row">
                <div class="parameter-name">Power Factor</div>
                <div class="parameter-value">0.95</div>
            </div>

            <div class="parameter-row">
                <div class="parameter-name">Temperature</div>
                <div class="parameter-value">30 °C</div>
            </div>

            <div class="parameter-row">
                <div class="parameter-name">THD</div>
                <div class="parameter-value">2.0 %</div>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ---------------- STATISTICS ----------------

    st.markdown("""
    <div class="section">

        <div class="section-label">
            SYSTEM CAPABILITIES
        </div>

        <div class="section-title">
            Built around measurable electrical behaviour.
        </div>

        <div class="section-description">
            The system combines multiple electrical measurements
            with a trained classification model to provide an
            automated assessment of operating conditions.
        </div>

    </div>
    """, unsafe_allow_html=True)


    s1, s2, s3, s4 = st.columns(4)

    statistics = [
        ("91.67%", "MODEL ACCURACY"),
        ("4", "FAULT CLASSES"),
        ("6", "INPUT PARAMETERS"),
        ("100", "RANDOM FOREST ESTIMATORS")
    ]

    for column, (value, label) in zip(
        [s1, s2, s3, s4],
        statistics
    ):

        with column:

            st.markdown(f"""
            <div class="stat">

                <div class="stat-value">
                    {value}
                </div>

                <div class="stat-label">
                    {label}
                </div>

            </div>
            """, unsafe_allow_html=True)


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown("""
    <div class="section">

        <div class="section-label">
            WORKFLOW
        </div>

        <div class="section-title">
            How the system works
        </div>

        <div class="section-description">
            A four-stage process converts electrical measurements
            into a fault classification.
        </div>

    </div>
    """, unsafe_allow_html=True)


    p1, p2, p3, p4 = st.columns(4)


    processes = [

        (
            "01",
            "Measure",
            "Electrical operating parameters are provided as inputs to the system."
        ),

        (
            "02",
            "Pre-process",
            "Input values are transformed using the same scaling method used during training."
        ),

        (
            "03",
            "Classify",
            "The Random Forest model evaluates the processed measurements and identifies the operating condition."
        ),

        (
            "04",
            "Report",
            "The predicted condition and model confidence are presented to the user."
        )

    ]


    for column, (number, title, description) in zip(
        [p1, p2, p3, p4],
        processes
    ):

        with column:

            st.markdown(f"""
            <div class="process">

                <div class="process-number">
                    {number}
                </div>

                <div class="process-title">
                    {title}
                </div>

                <div class="process-description">
                    {description}
                </div>

            </div>
            """, unsafe_allow_html=True)


    # ========================================================
    # DETECTION CAPABILITY
    # ========================================================

    st.markdown("""
    <div class="section">

        <div class="section-label">
            DETECTION CAPABILITY
        </div>

        <div class="section-title">
            Electrical conditions covered
        </div>

    </div>
    """, unsafe_allow_html=True)


    conditions = [
        ("Normal operation",
         "Electrical measurements remain within the patterns learned by the model."),

        ("Overvoltage",
         "The system identifies operating conditions associated with elevated voltage."),

        ("Undervoltage",
         "The model identifies operating conditions associated with reduced voltage."),

        ("Overcurrent",
         "The system detects operating conditions associated with excessive current.")
    ]


    for title, description in conditions:

        st.markdown(f"""
        <div class="feature">

            <div class="feature-title">
                {title}
            </div>

            <div class="feature-text">
                {description}
            </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <div class="footer">
        ElectroGuard AI — Machine Learning based electrical fault classification
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOGIN
# ============================================================

elif st.session_state.page == "login":

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.1, 1])

    with center:

        st.markdown("""
        <div class="login-box">

            <div class="login-title">
                ElectroGuard AI
            </div>

            <div class="login-subtitle">
                Sign in to access the monitoring platform
            </div>

        </div>
        """, unsafe_allow_html=True)

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
            use_container_width=True
        ):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.page = "dashboard"
                st.rerun()

            else:

                st.error("Incorrect username or password.")

        st.caption(
            "Demo credentials: admin / admin123"
        )

        if st.button(
            "Return to homepage",
            use_container_width=True
        ):

            st.session_state.page = "home"
            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

elif st.session_state.page == "dashboard":

    with st.sidebar:

        st.markdown(
            "<h2>ElectroGuard AI</h2>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        dashboard_page = st.radio(
            "Platform",
            [
                "Monitoring",
                "Fault Prediction",
                "About"
            ]
        )

        st.markdown("---")

        if st.button(
            "Sign out",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.page = "home"
            st.rerun()


    # ========================================================
    # MONITORING
    # ========================================================

    if dashboard_page == "Monitoring":

        st.markdown("""
        <div class="dashboard-header">

            <div class="dashboard-title">
                System monitoring
            </div>

            <div class="dashboard-subtitle">
                Current electrical operating parameters
            </div>

        </div>
        """, unsafe_allow_html=True)


        values = [
            ("Voltage", "230 V"),
            ("Current", "5.0 A"),
            ("Frequency", "50 Hz"),
            ("Power Factor", "0.95"),
            ("Temperature", "30 °C"),
            ("THD", "2.0 %")
        ]


        row1 = st.columns(3)

        for column, (label, value) in zip(
            row1,
            values[:3]
        ):

            with column:

                st.markdown(f"""
                <div class="monitor-card">

                    <div class="monitor-label">
                        {label}
                    </div>

                    <div class="monitor-value">
                        {value}
                    </div>

                </div>
                """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)


        row2 = st.columns(3)

        for column, (label, value) in zip(
            row2,
            values[3:]
        ):

            with column:

                st.markdown(f"""
                <div class="monitor-card">

                    <div class="monitor-label">
                        {label}
                    </div>

                    <div class="monitor-value">
                        {value}
                    </div>

                </div>
                """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown("""
        <div class="normal-result">

            <div class="normal-title">
                System ready
            </div>

            <p>
                The monitoring interface is ready for
                machine-learning based fault analysis.
            </p>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # FAULT PREDICTION
    # ========================================================

    elif dashboard_page == "Fault Prediction":

        st.markdown("""
        <div class="dashboard-header">

            <div class="dashboard-title">
                Fault prediction
            </div>

            <div class="dashboard-subtitle">
                Enter electrical measurements for model classification
            </div>

        </div>
        """, unsafe_allow_html=True)


        col1, col2 = st.columns(2)


        with col1:

            voltage = st.number_input(
                "Voltage (V)",
                min_value=0.0,
                max_value=500.0,
                value=230.0
            )

            current = st.number_input(
                "Current (A)",
                min_value=0.0,
                max_value=50.0,
                value=5.0
            )

            frequency = st.number_input(
                "Frequency (Hz)",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )


        with col2:

            power_factor = st.number_input(
                "Power Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.95
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-50.0,
                max_value=150.0,
                value=30.0
            )

            thd = st.number_input(
                "THD (%)",
                min_value=0.0,
                max_value=100.0,
                value=2.0
            )


        st.markdown("<br>", unsafe_allow_html=True)


        if st.button(
            "Run fault analysis",
            use_container_width=True
        ):

            input_data = pd.DataFrame(
                [[
                    voltage,
                    current,
                    frequency,
                    power_factor,
                    temperature,
                    thd
                ]],
                columns=[
                    "Voltage",
                    "Current",
                    "Frequency",
                    "Power_Factor",
                    "Temperature",
                    "THD"
                ]
            )


            input_scaled = scaler.transform(input_data)


            prediction = model.predict(
                input_scaled
            )[0]


            probabilities = model.predict_proba(
                input_scaled
            )[0]


            confidence = max(probabilities) * 100


            st.markdown("---")


            if prediction == "Normal":

                st.markdown(f"""
                <div class="normal-result">

                    <div class="normal-title">
                        Normal operating condition
                    </div>

                    <p>
                        No fault condition was classified by the model.
                    </p>

                    <strong>
                        Model confidence: {confidence:.2f}%
                    </strong>

                </div>
                """, unsafe_allow_html=True)


            else:

                st.markdown(f"""
                <div class="fault-result">

                    <div class="fault-title">
                        Fault condition detected
                    </div>

                    <p>
                        Classified condition:
                        <strong>{prediction}</strong>
                    </p>

                    <strong>
                        Model confidence: {confidence:.2f}%
                    </strong>

                </div>
                """, unsafe_allow_html=True)


            st.markdown("<br>", unsafe_allow_html=True)


            st.subheader("Measurement summary")

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ABOUT
    # ========================================================

    elif dashboard_page == "About":

        st.markdown("""
        <div class="dashboard-header">

            <div class="dashboard-title">
                About the system
            </div>

            <div class="dashboard-subtitle">
                Technical overview of ElectroGuard AI
            </div>

        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div class="system-panel">

            <h3>
                Machine Learning based electrical fault classification
            </h3>

            <p style="color:#64748b;line-height:1.8;">

            ElectroGuard AI is a prototype system that uses
            electrical operating parameters to classify the
            condition of an electrical system.

            The model was trained using a Random Forest classifier
            and evaluated using a held-out test dataset.

            </p>

            <hr>

            <p>
                <strong>Algorithm:</strong> Random Forest Classifier
            </p>

            <p>
                <strong>Pre-processing:</strong> StandardScaler
            </p>

            <p>
                <strong>Prototype accuracy:</strong> 91.67%
            </p>

            <p>
                <strong>Input parameters:</strong>
                Voltage, Current, Frequency, Power Factor,
                Temperature and THD
            </p>

            <p>
                <strong>Output classes:</strong>
                Normal, Overvoltage, Undervoltage and Overcurrent
            </p>

        </div>
        """, unsafe_allow_html=True)
