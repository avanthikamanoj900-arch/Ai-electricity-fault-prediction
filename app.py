import streamlit as st
import pandas as pd
import joblib

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
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 180, 255, 0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(0, 255, 170, 0.08), transparent 30%),
        #07111f;
    color: #f5f7fa;
}

/* Remove Streamlit header */
header {
    visibility: hidden;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Navigation */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 5px 30px 5px;
}

.logo {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
}

.logo span {
    color: #00d9ff;
}

.nav-text {
    color: #91a4b8;
    font-size: 14px;
}

/* Hero */
.hero {
    padding: 75px 0 60px 0;
}

.badge {
    display: inline-block;
    padding: 9px 16px;
    border-radius: 30px;
    background: rgba(0, 217, 255, 0.10);
    border: 1px solid rgba(0, 217, 255, 0.25);
    color: #00d9ff;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.hero-title {
    font-size: 68px;
    line-height: 1.03;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 25px;
    color: #ffffff;
}

.hero-title span {
    color: #00d9ff;
}

.hero-description {
    color: #91a4b8;
    font-size: 19px;
    line-height: 1.7;
    max-width: 650px;
}

/* Hero visual */
.energy-card {
    background: linear-gradient(
        145deg,
        rgba(17, 35, 58, 0.95),
        rgba(7, 17, 31, 0.95)
    );
    border: 1px solid rgba(0, 217, 255, 0.20);
    border-radius: 28px;
    padding: 35px;
    box-shadow: 0 20px 70px rgba(0,0,0,0.35);
}

.energy-circle {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    margin: 10px auto 30px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        radial-gradient(
            circle,
            rgba(0,217,255,0.25),
            rgba(0,217,255,0.04) 45%,
            transparent 70%
        );
    border: 1px solid rgba(0,217,255,0.35);
    box-shadow:
        0 0 50px rgba(0,217,255,0.18),
        inset 0 0 30px rgba(0,217,255,0.08);
}

.energy-icon {
    font-size: 82px;
}

.status-online {
    text-align: center;
    color: #39e58c;
    font-weight: 700;
    font-size: 14px;
}

/* Cards */
.feature-card {
    background: rgba(14, 29, 48, 0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 28px;
    height: 170px;
    transition: 0.2s;
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: white;
}

.feature-text {
    color: #8295aa;
    font-size: 13px;
    line-height: 1.6;
}

/* Statistics */
.stat-card {
    background: rgba(14,29,48,0.80);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #00d9ff;
}

.stat-label {
    color: #8193a8;
    font-size: 12px;
    margin-top: 5px;
}

/* Section heading */
.section-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-top: 70px;
    margin-bottom: 10px;
}

.section-subtitle {
    color: #8295aa;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Dashboard */
.dashboard-card {
    background: rgba(14,29,48,0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 25px;
}

.dashboard-number {
    font-size: 31px;
    font-weight: 800;
    color: #ffffff;
}

.dashboard-label {
    color: #8193a8;
    font-size: 13px;
}

/* Normal */
.normal-result {
    background: linear-gradient(
        135deg,
        rgba(32, 205, 122, 0.15),
        rgba(32, 205, 122, 0.04)
    );
    border: 1px solid rgba(57,229,140,0.35);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
}

/* Fault */
.fault-result {
    background: linear-gradient(
        135deg,
        rgba(255, 70, 70, 0.16),
        rgba(255, 70, 70, 0.04)
    );
    border: 1px solid rgba(255,80,80,0.40);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
}

/* Login */
.login-container {
    background: rgba(14,29,48,0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 45px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
}

.login-logo {
    text-align: center;
    font-size: 60px;
}

.login-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
}

.login-subtitle {
    text-align: center;
    color: #8193a8;
    margin-bottom: 30px;
}

/* Footer */
.footer {
    text-align: center;
    color: #61758a;
    padding: 60px 0 20px 0;
    font-size: 13px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(0,217,255,0.3);
    background: linear-gradient(
        90deg,
        #008fc7,
        #00b8d9
    );
    color: white;
    font-weight: 700;
    min-height: 48px;
}

.stButton > button:hover {
    border-color: #00d9ff;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background: #0b192b;
    color: white;
    border: 1px solid #263b53;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #081522;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # Navigation
    st.markdown("""
    <div class="navbar">
        <div class="logo">⚡ Electro<span>Guard</span> AI</div>
        <div class="nav-text">
            INTELLIGENT ELECTRICAL MONITORING
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    col1, col2 = st.columns([1.25, 0.75])

    with col1:

        st.markdown("""
        <div class="hero">

        <div class="badge">
        ⚡ AI-POWERED ELECTRICAL INTELLIGENCE
        </div>

        <div class="hero-title">
        Predict faults.<br>
        <span>Prevent failures.</span>
        </div>

        <div class="hero-description">
        ElectroGuard AI analyzes electrical parameters using
        Machine Learning to detect abnormal conditions before
        they become critical.
        </div>

        <br>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🚀 START MONITORING",
            use_container_width=False
        ):
            st.session_state.page = "login"
            st.rerun()

    with col2:

        st.markdown("""
        <div class="energy-card">

            <div class="energy-circle">
                <div class="energy-icon">⚡</div>
            </div>

            <div class="status-online">
                ● SYSTEM ONLINE
            </div>

            <br>

            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div style="color:#8193a8;font-size:12px;">
                    VOLTAGE
                    </div>
                    <b style="font-size:25px;">230 V</b>
                </div>

                <div>
                    <div style="color:#8193a8;font-size:12px;">
                    CURRENT
                    </div>
                    <b style="font-size:25px;">5.0 A</b>
                </div>
            </div>

            <br>

            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div style="color:#8193a8;font-size:12px;">
                    FREQUENCY
                    </div>
                    <b style="font-size:25px;">50 Hz</b>
                </div>

                <div>
                    <div style="color:#8193a8;font-size:12px;">
                    POWER FACTOR
                    </div>
                    <b style="font-size:25px;">0.95</b>
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

    # Statistics
    st.markdown(
        "<div class='section-title'>Built for intelligent protection.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-subtitle'>"
        "A machine-learning driven approach to electrical fault detection."
        "</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("91.67%", "MODEL ACCURACY"),
        ("4", "FAULT CLASSES"),
        ("6", "ELECTRICAL PARAMETERS"),
        ("AI", "PREDICTION ENGINE")
    ]

    for col, (number, label) in zip(
        [c1, c2, c3, c4],
        stats
    ):

        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Features
    st.markdown(
        "<div class='section-title'>Why ElectroGuard AI?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-subtitle'>"
        "From raw electrical measurements to actionable intelligence."
        "</div>",
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    features = [
        (
            "⚡",
            "Early Fault Detection",
            "Identify abnormal electrical conditions before they escalate."
        ),
        (
            "🤖",
            "AI Classification",
            "Random Forest Machine Learning automatically classifies faults."
        ),
        (
            "📊",
            "Multi-Parameter Analysis",
            "Analyze voltage, current, frequency, power factor, temperature and THD."
        )
    ]

    for col, (icon, title, text) in zip(
        [f1, f2, f3],
        features
    ):

        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # How it works
    st.markdown(
        "<div class='section-title'>How it works</div>",
        unsafe_allow_html=True
    )

    steps = [
        ("01", "Measure", "Collect electrical parameters."),
        ("02", "Analyze", "Preprocess the input data."),
        ("03", "Predict", "AI model identifies the condition."),
        ("04", "Respond", "Display the fault and confidence.")
    ]

    s1, s2, s3, s4 = st.columns(4)

    for col, (num, title, text) in zip(
        [s1, s2, s3, s4],
        steps
    ):

        with col:
            st.markdown(f"""
            <div class="dashboard-card">

                <div style="
                    color:#00d9ff;
                    font-size:13px;
                    font-weight:800;
                ">
                    {num}
                </div>

                <h3 style="color:white;">
                    {title}
                </h3>

                <div style="
                    color:#8193a8;
                    font-size:13px;
                    line-height:1.5;
                ">
                    {text}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        ⚡ ElectroGuard AI &nbsp; | &nbsp;
        Intelligent Electrical Fault Prediction
        <br><br>
        Built with Python • Scikit-learn • Random Forest • Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOGIN PAGE
# ============================================================

elif st.session_state.page == "login":

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.2, 1])

    with center:

        st.markdown("""
        <div class="login-container">

            <div class="login-logo">⚡</div>

            <div class="login-title">
                Welcome to ElectroGuard
            </div>

            <div class="login-subtitle">
                Secure access to your AI monitoring dashboard
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
            "🔐 LOGIN TO DASHBOARD",
            use_container_width=True
        ):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.page = "dashboard"
                st.rerun()

            else:

                st.error("Invalid username or password.")

        if st.button(
            "← Back to Home",
            use_container_width=True
        ):

            st.session_state.page = "home"
            st.rerun()

        st.info(
            "Demo credentials: admin / admin123"
        )


# ============================================================
# DASHBOARD
# ============================================================

elif st.session_state.page == "dashboard":

    # Sidebar
    with st.sidebar:

        st.markdown("""
        <div style="
            font-size:24px;
            font-weight:800;
            color:white;
            padding:10px 0 20px 0;
        ">
            ⚡ ElectroGuard
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navigation")

        dashboard_page = st.radio(
            "",
            [
                "📊 Monitoring",
                "🔍 Fault Prediction",
                "ℹ️ About"
            ]
        )

        st.markdown("---")

        if st.button("🚪 Logout"):

            st.session_state.logged_in = False
            st.session_state.page = "home"
            st.rerun()

    # Monitoring
    if dashboard_page == "📊 Monitoring":

        st.markdown(
            "<div class='title'>Live Monitoring</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='subtitle'>"
            "Electrical system overview and operating parameters"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        metrics = [
            ("⚡", "Voltage", "230 V"),
            ("🔌", "Current", "5.0 A"),
            ("〰️", "Frequency", "50 Hz")
        ]

        for col, (icon, label, value) in zip(
            [c1, c2, c3],
            metrics
        ):

            with col:
                st.markdown(f"""
                <div class="dashboard-card">

                    <div style="font-size:28px;">
                        {icon}
                    </div>

                    <div class="dashboard-label">
                        {label}
                    </div>

                    <div class="dashboard-number">
                        {value}
                    </div>

                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)

        metrics2 = [
            ("📈", "Power Factor", "0.95"),
            ("🌡️", "Temperature", "30 °C"),
            ("〰️", "THD", "2 %")
        ]

        for col, (icon, label, value) in zip(
            [c4, c5, c6],
            metrics2
        ):

            with col:
                st.markdown(f"""
                <div class="dashboard-card">

                    <div style="font-size:28px;">
                        {icon}
                    </div>

                    <div class="dashboard-label">
                        {label}
                    </div>

                    <div class="dashboard-number">
                        {value}
                    </div>

                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="normal-result">

            <h1 style="color:#39e58c;">
                ● SYSTEM ONLINE
            </h1>

            <p style="color:#9bb0c4;">
                Monitoring system is ready for AI-based
                electrical fault analysis.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # FAULT PREDICTION
    # ========================================================

    elif dashboard_page == "🔍 Fault Prediction":

        st.markdown(
            "<div class='title'>AI Fault Prediction</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='subtitle'>"
            "Enter electrical parameters to classify the system condition."
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:

            voltage = st.number_input(
                "⚡ Voltage (V)",
                min_value=0.0,
                max_value=500.0,
                value=230.0
            )

            current = st.number_input(
                "🔌 Current (A)",
                min_value=0.0,
                max_value=50.0,
                value=5.0
            )

            frequency = st.number_input(
                "〽️ Frequency (Hz)",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )

        with c2:

            power_factor = st.number_input(
                "📈 Power Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.95
            )

            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-50.0,
                max_value=150.0,
                value=30.0
            )

            thd = st.number_input(
                "〰️ THD (%)",
                min_value=0.0,
                max_value=100.0,
                value=2.0
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "⚡ ANALYZE ELECTRICAL CONDITION",
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

            prediction = model.predict(input_scaled)[0]

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            confidence = max(probabilities) * 100

            st.markdown("---")

            if prediction == "Normal":

                st.markdown(f"""
                <div class="normal-result">

                    <div style="font-size:55px;">
                        ✓
                    </div>

                    <h1 style="color:#39e58c;">
                        SYSTEM NORMAL
                    </h1>

                    <h2 style="color:white;">
                        Confidence: {confidence:.2f}%
                    </h2>

                    <p style="color:#9bb0c4;">
                        No significant electrical fault detected.
                    </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="fault-result">

                    <div style="font-size:55px;">
                        ⚠
                    </div>

                    <h1 style="color:#ff6262;">
                        FAULT DETECTED
                    </h1>

                    <h2 style="color:white;">
                        {prediction}
                    </h2>

                    <h3 style="color:#ff9b9b;">
                        Confidence: {confidence:.2f}%
                    </h3>

                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("📋 Analyzed Parameters")

            st.dataframe(
                input_data,
                use_container_width=True
            )

    # ========================================================
    # ABOUT
    # ========================================================

    elif dashboard_page == "ℹ️ About":

        st.markdown(
            "<div class='title'>About ElectroGuard AI</div>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="dashboard-card">

        <h2 style="color:#00d9ff;">
        ⚡ AI-Based Electrical Fault Prediction
        </h2>

        <p style="color:#9bb0c4;line-height:1.8;">

        ElectroGuard AI is a Machine Learning based prototype
        designed to detect and classify electrical faults using
        multiple electrical operating parameters.

        </p>

        <h3 style="color:white;">Machine Learning Model</h3>

        <p style="color:#9bb0c4;">
        Random Forest Classifier
        </p>

        <h3 style="color:white;">Input Parameters</h3>

        <p style="color:#9bb0c4;">
        Voltage • Current • Frequency • Power Factor •
        Temperature • THD
        </p>

        <h3 style="color:white;">Fault Classes</h3>

        <p style="color:#9bb0c4;">
        Normal • Overvoltage • Undervoltage • Overcurrent
        </p>

        <h3 style="color:white;">Prototype Accuracy</h3>

        <p style="color:#00d9ff;font-size:28px;font-weight:800;">
        91.67%
        </p>

        </div>
        """, unsafe_allow_html=True)
