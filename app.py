Sure — here is a complete app.py you can run directly with Streamlit. It creates a cleaner, more human-designed electrical fault detection dashboard, including the redesigned How It Works section.

import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GridSense | Electrical Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background: #07121c;
    color: #e9f0f5;
}

/* Remove Streamlit top spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =====================================================
   NAVBAR
   ===================================================== */

.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 24px;
    border-bottom: 1px solid #1d2d3a;
    margin-bottom: 45px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 11px;
}

.logo-mark {
    width: 32px;
    height: 32px;
    background: #0b9bea;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 16px;
    border-radius: 6px;
}

.logo-name {
    font-size: 17px;
    font-weight: 700;
    color: #f3f7fa;
    letter-spacing: -0.3px;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 22px;
    color: #75899a;
    font-size: 12px;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 7px;
}

/* =====================================================
   HERO
   ===================================================== */

.hero {
    padding: 25px 0 65px;
    max-width: 950px;
}

.eyebrow {
    font-family: "IBM Plex Mono", monospace;
    color: #39a9ed;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-size: clamp(42px, 6vw, 72px);
    line-height: 1.02;
    letter-spacing: -3px;
    font-weight: 700;
    color: #f5f8fa;
    margin: 0 0 24px;
}

.hero-title span {
    color: #43b4f3;
}

.hero-text {
    color: #899baa;
    font-size: 16px;
    line-height: 1.7;
    max-width: 650px;
}

/* =====================================================
   METRICS
   ===================================================== */

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid #243442;
    border-bottom: 1px solid #243442;
    margin-bottom: 85px;
}

.metric {
    padding: 24px 25px;
    border-right: 1px solid #243442;
}

.metric:last-child {
    border-right: none;
}

.metric-value {
    font-size: 27px;
    font-weight: 600;
    color: #f3f7fa;
    margin-bottom: 5px;
}

.metric-label {
    font-family: "IBM Plex Mono", monospace;
    color: #64798a;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .7px;
}

/* =====================================================
   SECTION HEADER
   ===================================================== */

.section {
    margin-bottom: 90px;
}

.section-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid #243442;
    padding-bottom: 17px;
    margin-bottom: 28px;
}

.section-title {
    font-size: 29px;
    font-weight: 600;
    color: #f3f7fa;
    letter-spacing: -.7px;
}

.section-description {
    color: #718595;
    font-size: 13px;
    line-height: 1.6;
    max-width: 430px;
    text-align: right;
}

/* =====================================================
   HOW IT WORKS
   ===================================================== */

.process {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid #263846;
    background: #091722;
}

.process-item {
    min-height: 265px;
    padding: 27px 25px;
    border-right: 1px solid #263846;
    position: relative;
    transition: background .2s ease;
}

.process-item:last-child {
    border-right: none;
}

.process-item:hover {
    background: #0d1d29;
}

.process-number {
    font-family: "IBM Plex Mono", monospace;
    color: #3aa9eb;
    font-size: 11px;
    letter-spacing: .8px;
}

.process-line {
    width: 32px;
    height: 1px;
    background: #315267;
    margin: 50px 0 22px;
}

.process-heading {
    color: #f1f5f8;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 11px;
}

.process-description {
    color: #7c8f9e;
    font-size: 13px;
    line-height: 1.7;
    max-width: 240px;
}

.process-label {
    position: absolute;
    bottom: 24px;
    left: 25px;
    color: #405463;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* =====================================================
   DASHBOARD
   ===================================================== */

.dashboard {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 20px;
}

.panel {
    border: 1px solid #263846;
    background: #091722;
    padding: 25px;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.panel-title {
    font-size: 14px;
    font-weight: 600;
    color: #e8eef2;
}

.panel-tag {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: #3aa9eb;
    border: 1px solid #21475d;
    padding: 5px 8px;
}

/* =====================================================
   FAULT TABLE
   ===================================================== */

.fault {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 17px 0;
    border-top: 1px solid #1c2c38;
}

.fault:last-child {
    border-bottom: 1px solid #1c2c38;
}

.fault-name {
    font-size: 13px;
    color: #dbe4e9;
}

.fault-meta {
    color: #607585;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    margin-top: 4px;
}

.badge {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    padding: 5px 8px;
}

.badge-high {
    color: #fb7185;
    background: #26151b;
    border: 1px solid #51212c;
}

.badge-medium {
    color: #fbbf24;
    background: #251f12;
    border: 1px solid #514317;
}

.badge-low {
    color: #34d399;
    background: #10251e;
    border: 1px solid #1c4a3b;
}

/* =====================================================
   SIGNAL CARDS
   ===================================================== */

.signal-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}

.signal {
    background: #0c1b27;
    border: 1px solid #20323f;
    padding: 17px;
}

.signal-name {
    font-family: "IBM Plex Mono", monospace;
    color: #617788;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.signal-value {
    font-size: 22px;
    font-weight: 600;
    color: #edf3f6;
    margin-top: 9px;
}

.signal-unit {
    font-size: 11px;
    color: #637887;
    margin-left: 3px;
}

/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    border-top: 1px solid #243442;
    padding-top: 22px;
    color: #526775;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    display: flex;
    justify-content: space-between;
}

/* =====================================================
   STREAMLIT BUTTON
   ===================================================== */

.stButton > button {
    background: #0d9be8;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #18a9f5;
    color: white;
}

/* =====================================================
   RESPONSIVE
   ===================================================== */

@media (max-width: 900px) {

    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .metric:nth-child(2) {
        border-right: none;
    }

    .metric:nth-child(-n+2) {
        border-bottom: 1px solid #243442;
    }

    .process {
        grid-template-columns: repeat(2, 1fr);
    }

    .process-item:nth-child(2) {
        border-right: none;
    }

    .process-item:nth-child(-n+2) {
        border-bottom: 1px solid #263846;
    }

    .dashboard {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 600px) {

    .hero-title {
        letter-spacing: -2px;
    }

    .nav-right {
        display: none;
    }

    .metrics {
        grid-template-columns: 1fr 1fr;
    }

    .process {
        grid-template-columns: 1fr;
    }

    .process-item {
        border-right: none !important;
        border-bottom: 1px solid #263846;
    }

    .process-item:last-child {
        border-bottom: none;
    }

    .section-head {
        display: block;
    }

    .section-description {
        text-align: left;
        margin-top: 10px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

st.markdown("""
<div class="nav">

    <div class="logo">
        <div class="logo-mark">⚡</div>
        <div class="logo-name">GRIDSENSE</div>
    </div>

    <div class="nav-right">
        <div>
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>
        <div>v2.4.1</div>
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div class="eyebrow">
        Electrical intelligence / predictive maintenance
    </div>

    <div class="hero-title">
        Detect problems<br>
        <span>before they fail.</span>
    </div>

    <div class="hero-text">
        Monitor electrical behavior, identify abnormal patterns,
        and turn raw machine data into actionable maintenance
        intelligence.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================

st.markdown("""
<div class="metrics">

    <div class="metric">
        <div class="metric-value">24/7</div>
        <div class="metric-label">Continuous monitoring</div>
    </div>

    <div class="metric">
        <div class="metric-value">&lt; 2s</div>
        <div class="metric-label">Detection latency</div>
    </div>

    <div class="metric">
        <div class="metric-value">98.4%</div>
        <div class="metric-label">Model confidence</div>
    </div>

    <div class="metric">
        <div class="metric-value">12.8k</div>
        <div class="metric-label">Signals processed</div>
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("""
<div class="section">

    <div class="section-head">

        <div class="section-title">
            How it works
        </div>

        <div class="section-description">
            Four stages transform electrical measurements
            into clear maintenance decisions.
        </div>

    </div>

    <div class="process">

        <div class="process-item">
            <div class="process-number">01 / INPUT</div>

            <div class="process-line"></div>

            <div class="process-heading">
                Measure
            </div>

            <div class="process-description">
                Capture voltage, current, power and other
                electrical parameters from connected equipment.
            </div>

            <div class="process-label">
                Data acquisition
            </div>
        </div>


        <div class="process-item">
            <div class="process-number">02 / PROCESS</div>

            <div class="process-line"></div>

            <div class="process-heading">
                Analyze
            </div>

            <div class="process-description">
                Clean and normalize incoming signals before
                extracting the patterns that matter.
            </div>

            <div class="process-label">
                Signal processing
            </div>
        </div>


        <div class="process-item">
            <div class="process-number">03 / INTELLIGENCE</div>

            <div class="process-line"></div>

            <div class="process-heading">
                Predict
            </div>

            <div class="process-description">
                Identify abnormal behavior and estimate
                potential equipment faults.
            </div>

            <div class="process-label">
                Fault detection
            </div>
        </div>


        <div class="process-item">
            <div class="process-number">04 / ACTION</div>

            <div class="process-line"></div>

            <div class="process-heading">
                Respond
            </div>

            <div class="process-description">
                Surface the detected fault with clear information
                so maintenance teams can act quickly.
            </div>

            <div class="process-label">
                Decision support
            </div>
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# LIVE MONITORING
# =========================================================

st.markdown("""
<div class="section">

    <div class="section-head">

        <div class="section-title">
            Live monitoring
        </div>

        <div class="section-description">
            Current electrical conditions across the monitored
            equipment.
        </div>

    </div>

    <div class="dashboard">

        <div class="panel">

            <div class="panel-header">
                <div class="panel-title">Detected conditions</div>
                <div class="panel-tag">LIVE</div>
            </div>

            <div class="fault">

                <div>
                    <div class="fault-name">
                        Motor 04 — Current imbalance
                    </div>
                    <div class="fault-meta">
                        DETECTED 2 MIN AGO
                    </div>
                </div>

                <div class="badge badge-high">
                    HIGH
                </div>

            </div>


            <div class="fault">

                <div>
                    <div class="fault-name">
                        Pump 07 — Voltage fluctuation
                    </div>
                    <div class="fault-meta">
                        DETECTED 14 MIN AGO
                    </div>
                </div>

                <div class="badge badge-medium">
                    MEDIUM
                </div>

            </div>


            <div class="fault">

                <div>
                    <div class="fault-name">
                        Compressor 02 — Normal
                    </div>
                    <div class="fault-meta">
                        LAST CHECK 32 SEC AGO
                    </div>
                </div>

                <div class="badge badge-low">
                    NORMAL
                </div>

            </div>

        </div>


        <div class="panel">

            <div class="panel-header">
                <div class="panel-title">Electrical signals</div>
                <div class="panel-tag">NODE 04</div>
            </div>

            <div class="signal-grid">

                <div class="signal">
                    <div class="signal-name">Voltage</div>
                    <div class="signal-value">
                        412<span class="signal-unit">V</span>
                    </div>
                </div>

                <div class="signal">
                    <div class="signal-name">Current</div>
                    <div class="signal-value">
                        18.6<span class="signal-unit">A</span>
                    </div>
                </div>

                <div class="signal">
                    <div class="signal-name">Power</div>
                    <div class="signal-value">
                        7.4<span class="signal-unit">kW</span>
                    </div>
                </div>

                <div class="signal">
                    <div class="signal-name">Frequency</div>
                    <div class="signal-value">
                        49.9<span class="signal-unit">Hz</span>
                    </div>
                </div>

            </div>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# OPTIONAL DATA / DEMO SECTION
# =========================================================

st.markdown("""
<div class="section">

    <div class="section-head">

        <div class="section-title">
            Signal overview
        </div>

        <div class="section-description">
            Recent electrical measurements from the selected
            monitoring node.
        </div>

    </div>
</div>
""", unsafe_allow_html=True)


# Generate demo data
np.random.seed(42)

time = pd.date_range(
    end=pd.Timestamp.now(),
    periods=50,
    freq="min"
)

voltage = 410 + np.random.normal(0, 2.5, 50)
current = 18 + np.random.normal(0, 0.7, 50)
power = voltage * current / 1000

chart_data = pd.DataFrame(
    {
        "Voltage (V)": voltage,
        "Current (A)": current,
        "Power (kW)": power
    },
    index=time
)

st.line_chart(
    chart_data,
    height=300,
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <div>
        GRIDSENSE / INDUSTRIAL MONITORING
    </div>

    <div>
        ALL SYSTEMS OPERATIONAL
    </div>

</div>
""", unsafe_allow_html=True)

Run it
Save it as:

app.py
