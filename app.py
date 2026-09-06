import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ElectroGuard AI",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
}

.login-box {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 5px 25px rgba(0,0,0,0.08);
    text-align: center;
}

.title {
    font-size: 42px;
    font-weight: 800;
    color: #12355b;
}

.subtitle {
    font-size: 18px;
    color: #607080;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.metric-title {
    font-size: 16px;
    color: #687684;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #12355b;
}

.normal-box {
    background: #e8f8ef;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #42b883;
}

.fault-box {
    background: #fff0f0;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #e05252;
}

.footer {
    text-align: center;
    color: #7a8793;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


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
# LOGIN SYSTEM
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown("""
        <div class="login-box">

        <div style="font-size:60px;">⚡</div>

        <div class="title">
        ElectroGuard AI
        </div>

        <div class="subtitle">
        AI-Powered Electrical Fault Detection
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Login")

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button("🚀 LOGIN", use_container_width=True):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error("❌ Invalid username or password")

        st.info(
            "Demo Login → Username: admin | Password: admin123"
        )

    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ ElectroGuard AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Fault Prediction",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False
    st.rerun()


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        "<div class='title'>⚡ ElectroGuard AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "AI-Based Electrical Fault Monitoring System"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("""
        <div class="card">

        <div class="metric-title">
        Voltage
        </div>

        <div class="metric-value">
        230 V
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <div class="metric-title">
        Current
        </div>

        <div class="metric-value">
        5 A
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

        <div class="metric-title">
        Frequency
        </div>

        <div class="metric-value">
        50 Hz
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="card">

        <div class="metric-title">
        Power Factor
        </div>

        <div class="metric-value">
        0.95
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="normal-box">

    <h2>🟢 SYSTEM READY</h2>

    <p>
    The AI fault detection system is ready to analyze
    electrical parameters.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Go to 🔍 Fault Prediction from the sidebar "
        "to analyze electrical parameters."
    )


# --------------------------------------------------
# FAULT PREDICTION
# --------------------------------------------------

elif page == "🔍 Fault Prediction":

    st.markdown(
        "<div class='title'>🔍 Fault Prediction</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Enter the electrical parameters and let the AI model "
        "identify the system condition."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

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
        "🔍 PREDICT ELECTRICAL FAULT",
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

        probabilities = model.predict_proba(input_scaled)[0]

        confidence = max(probabilities) * 100

        st.markdown("---")

        st.subheader("🤖 AI Prediction")

        if prediction == "Normal":

            st.markdown(f"""
            <div class="normal-box">

            <h1>🟢 NORMAL SYSTEM</h1>

            <h3>Confidence: {confidence:.2f}%</h3>

            <p>
            The electrical parameters appear to be
            within the normal operating condition.
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="fault-box">

            <h1>🚨 FAULT DETECTED</h1>

            <h2>{prediction}</h2>

            <h3>Confidence: {confidence:.2f}%</h3>

            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📋 Input Parameters")

        st.dataframe(
            input_data,
            use_container_width=True
        )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "ℹ️ About":

    st.markdown(
        "<div class='title'>ℹ️ About ElectroGuard AI</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h2>⚡ AI-Based Electrical Fault Prediction</h2>

    <p>
    ElectroGuard AI is a Machine Learning based electrical
    fault detection system designed to identify abnormal
    electrical operating conditions.
    </p>

    <h3>🧠 Machine Learning Model</h3>

    <p>
    Random Forest Classifier
    </p>

    <h3>📊 Input Parameters</h3>

    <ul>
        <li>Voltage</li>
        <li>Current</li>
        <li>Frequency</li>
        <li>Power Factor</li>
        <li>Temperature</li>
        <li>Total Harmonic Distortion</li>
    </ul>

    <h3>🚨 Detectable Conditions</h3>

    <ul>
        <li>Normal</li>
        <li>Overvoltage</li>
        <li>Undervoltage</li>
        <li>Overcurrent</li>
    </ul>

    <h3>📈 Prototype Accuracy</h3>

    <p>
    Approximately <b>91.67%</b>
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">

    ⚡ ElectroGuard AI | AI-Based Electrical Fault Prediction

    </div>
    """, unsafe_allow_html=True)
