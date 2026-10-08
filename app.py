# === Importing Libraries ==============================

import sys
import os
import time

import numpy as np
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# === Page Config ==============================

st.set_page_config(
    page_title="LaptiQ — Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# === Loading Model ==============================

@st.cache_resource
def load_model():
    return joblib.load("model/Lapti_Q.pkl")

model = load_model()


# === Constants & Session State ==============================

STORAGE_MAP = {
    "256 GB": 256,
    "512 GB": 512,
    "1 TB":  1024,
    "2 TB":  2048,
    "4 TB":  4096,
    "8 TB":  8192,
}

if "page" not in st.session_state:
    st.session_state["page"] = "home"


# === Helper Functions ==============================

def indian_format(price: int) -> str:
    s = str(price)
    if len(s) <= 3:
        return s
    last3  = s[-3:]
    rest   = s[:-3]
    groups = []
    while rest:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups = [g for g in reversed(groups) if g]
    return ",".join(groups) + "," + last3


def calculate_ppi(resolution: str, screen_size: float) -> int:
    w, h = map(int, resolution.split("x"))
    return round(np.sqrt(w**2 + h**2) / screen_size)


# === CSS ==============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background: #0f0f1a; }

.block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 1100px;
}

/* global button style */
div.stButton > button {
    width: 100%;
    padding: 0.9rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #0f0f1a;
    background: linear-gradient(135deg, #23d5ab, #1cc49e);
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
    text-transform: uppercase;
    box-shadow: 0 2px 12px rgba(196,181,253,0.25);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(196,181,253,0.4);
}

div.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(196,181,253,0.2);
}

/* primary action button */
div.stButton > button[kind="primary"] {
    background: #93C5FD !important;
    color: #0f0f1a !important;
    font-weight: 700 !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #bfdbfe !important;
    color: #0f0f1a !important;
}

/* sidebar nav buttons */
section[data-testid="stSidebar"] div.stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #a0a0b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0.2px !important;
    box-shadow: none !important;
    padding: 0.6rem 0.75rem !important;
    min-height: 40px !important;
    margin-bottom: 0.15rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #e0e0f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* number input step buttons */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    color: #ffffff !important;
    border-color: rgba(164, 99, 242, 0.3) !important;
    background: rgba(164, 99, 242, 0.08) !important;
    transition: all 0.2s ease !important;
}

button[data-testid="stNumberInputStepUp"]:hover,
button[data-testid="stNumberInputStepDown"]:hover {
    background: rgba(164, 99, 242, 0.18) !important;
    border-color: rgba(164, 99, 242, 0.5) !important;
    color: #c4a0ff !important;
}

button[data-testid="stNumberInputStepUp"]:active,
button[data-testid="stNumberInputStepDown"]:active {
    background: rgba(164, 99, 242, 0.28) !important;
    color: #d4b8ff !important;
}

/* ── Animations ───────────────────────────────────── */

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Price Result Card ────────────────────────────── */

.result-wrap {
    animation: fadeUp 0.5s ease forwards;
    opacity: 0;
}

.result-box {
    background: linear-gradient(135deg, rgba(35,213,171,0.1), rgba(164,99,242,0.08));
    border: 1px solid rgba(35,213,171,0.2);
    border-radius: 16px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 520px;
    backdrop-filter: blur(10px);
}

.result-label {
    font-size: 0.85rem;
    color: #7c7c90;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 0.4rem;
}

.result-price {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #23d5ab, #a463f2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-subprice {
    font-size: 0.95rem;
    font-weight: 500;
    color: #6a6a82;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
}

/* ── Loading Spinner ──────────────────────────────── */

.spin-dot {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(35,213,171,0.12);
    border-top: 3px solid #23d5ab;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-bottom: 0.8rem;
}

/* ── Sidebar ──────────────────────────────────────── */

section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1rem !important;
}

.sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0.6rem 0.5rem;
}

/* ── Hero Section ─────────────────────────────────── */

.hero-section {
    position: relative;
    min-height: 340px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2.5rem 1.2rem 2rem;
    background: #0f0f1a;
    background-image: radial-gradient(ellipse 70% 60% at 50% 50%, rgba(35,213,171,0.06) 0%, transparent 70%);
    overflow: hidden;
}

.hero-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #23d5ab;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0s;
}

.hero-heading { opacity: 1; }

.hero-line-1 {
    font-size: clamp(2.2rem, 6vw, 4.5rem);
    font-weight: 300;
    color: #8a8aa0;
    line-height: 1;
    margin: 0;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.1s;
}

.hero-line-2 {
    font-size: clamp(2.2rem, 6vw, 4.5rem);
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    margin: 0.5rem 0 0 0;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.2s;
}

.hero-subtitle {
    font-size: clamp(0.85rem, 2vw, 1.05rem);
    color: #7c7c90;
    margin-top: 0.5rem;
    max-width: 480px;
    text-align: center;
    line-height: 1.6;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.3s;
}

.hero-accent {
    width: 120px;
    height: 2px;
    margin: 1.8rem auto 0;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, #23d5ab, #a463f2, transparent);
    background-size: 200% 100%;
    animation: fadeUp 0.6s ease forwards, shimmer 3s ease infinite;
    animation-delay: 0.5s, 0.5s;
    opacity: 0;
}

/* ── Form Section ─────────────────────────────────── */

.fs-wrap {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.fs-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: rgba(120, 120, 250, 0.12);
    color: #8c8df0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.fs-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ffffff;
}

.fs-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1rem 0 1.5rem;
    width: 100%;
}

/* ── Model Info Page ──────────────────────────────── */

.section-title-wrap {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem;
}

.section-icon {
    display: flex;
    align-items: center;
    color: #4e8cff;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e0e0f0;
}

.section-divider {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* stat cards */
.sb-stat-new {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.7rem;
    display: flex;
    flex-direction: column;
}

.sb-stat-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}

.sb-stat-title { font-size: 0.9rem; color: #c0c0d0; font-weight: 500; }

.sb-stat-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.sb-stat-icon.blue   { background: rgba(78, 140, 255, 0.12);  color: #4e8cff; }
.sb-stat-icon.orange { background: rgba(234, 179, 8, 0.12);   color: #eab308; }
.sb-stat-icon.purple { background: rgba(180, 108, 248, 0.12); color: #b46cf8; }

.sb-stat-val-new { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; }
.sb-stat-val-new.blue   { color: #93C5FD; }
.sb-stat-val-new.orange { color: #FCD34D; }
.sb-stat-val-new.purple { color: #C4B5FD; }

.sb-stat-sub { font-size: 0.75rem; color: #7c7c90; }

/* step cards */
.mi-step-card {
    display: flex;
    align-items: center;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: background 0.2s ease;
}

.mi-step-card:hover { background: rgba(255,255,255,0.04); }

.mi-step-num {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #3b5bdb;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-right: 1.2rem;
}

.mi-step-content { flex: 1; }
.mi-step-title  { font-size: 0.95rem; color: #e0e0f0; font-weight: 500; margin-bottom: 0.25rem; }
.mi-step-desc   { font-size: 0.82rem; color: #8a8aa0; }
.mi-step-chevron { flex-shrink: 0; margin-left: 1rem; display: flex; align-items: center; }

/* ── Footer ───────────────────────────────────────── */

.footer-note {
    text-align: center;
    color: #4a4a5e;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding: 0.8rem 0;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-style: italic;
}

/* ── Input Widgets ────────────────────────────────── */

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e0e0f0 !important;
}

.stSlider > div > div > div { color: #23d5ab !important; }

.stSelectbox label, .stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label, .stCheckbox label, .stMultiSelect label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #c8c8dc !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

.stSelectbox [data-testid="stMarkdownContainer"],
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] .css-1dimb5e-singleValue,
.stSelectbox [data-baseweb="select"] > div > div > div,
.stSelectbox div[data-baseweb="select"] > div {
    color: #e0e0f0 !important;
}

[data-baseweb="popover"], [data-baseweb="popover"] ul,
[data-baseweb="menu"], [role="listbox"] {
    background: #1a1a2e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

[data-baseweb="popover"] li, [data-baseweb="menu"] li,
[role="option"], [data-baseweb="popover"] ul li, ul[role="listbox"] li {
    color: #d0d0e4 !important;
    background: transparent !important;
}

[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover,
[role="option"]:hover, li[aria-selected="true"],
[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(35,213,171,0.12) !important;
    color: #ffffff !important;
}

[data-baseweb="menu"] li[data-highlighted="true"],
[role="option"][data-highlighted] {
    background: rgba(35,213,171,0.15) !important;
    color: #ffffff !important;
}

.stTextInput input, .stNumberInput input { color: #e0e0f0 !important; }

.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #6a6a82 !important;
    opacity: 1 !important;
}

.stSelectbox svg,
.stSelectbox [data-baseweb="select"] svg {
    fill: #8a8aa0 !important;
    color: #8a8aa0 !important;
}

.stSelectbox [data-testid="stTooltipIcon"],
.stTextInput [data-testid="stTooltipIcon"],
.stNumberInput [data-testid="stTooltipIcon"] {
    color: #6a6a82 !important;
}

/* ── Lock Dark Background ─────────────────────────── */

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header[data-testid="stHeader"] {
    background: #0f0f1a !important;
}

[data-testid="stToolbar"], [data-testid="stStatusWidget"] { color: #7c7c90 !important; }

.streamlit-expanderHeader { color: #c8c8dc !important; }
.streamlit-expanderContent { color: #a0a0b8 !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Responsive: Tablets (≤768px) ─────────────────── */

@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        width: 240px !important;
        min-width: 240px !important;
        max-width: 240px !important;
    }
    .block-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-top: 1rem !important;
    }
    .hero-section { min-height: 240px; padding: 2rem 1rem; }
    .hero-line-1, .hero-line-2 { font-size: clamp(1.8rem, 8vw, 2.8rem); }
    .hero-subtitle { font-size: 0.82rem; }
    .hero-label { letter-spacing: 1.5px; }
    div.stButton > button { padding: 0.85rem 1.5rem; font-size: 1rem; border-radius: 10px; }
    .result-box      { padding: 1.3rem 1rem; margin: 1rem auto; border-radius: 12px; }
    .result-price    { font-size: 1.7rem !important; }
    .result-subprice { font-size: 0.82rem; }
    .result-label    { font-size: 0.75rem; letter-spacing: 1.2px; }
    .footer-note     { font-size: 0.7rem; margin-top: 1.5rem; }
}

/* ── Responsive: Small Phones (≤480px) ────────────── */

@media (max-width: 480px) {
    .result-price    { font-size: 1.35rem !important; }
    .result-subprice { font-size: 0.75rem; }
    div.stButton > button { font-size: 0.92rem; padding: 0.75rem 1rem; }
}

/* ── Responsive: iPads (769px–1024px) ─────────────── */

@media (min-width: 769px) and (max-width: 1024px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .result-price    { font-size: 2rem !important; }
    .result-subprice { font-size: 0.88rem; }
}
</style>
""", unsafe_allow_html=True)


# === Sidebar Navigation ==============================

with st.sidebar:

    nav_items = [
        ("home",       "Home"),
        ("model_info", "Model Info"),
    ]

    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.markdown('<div class="sb-divider">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4a4a5e;font-size:0.7rem;text-align:center;margin-top:0.5rem;">LaptiQ v1.0 · XGBoost</p>',
        unsafe_allow_html=True,
    )


# === PAGE: HOME ==============================

if st.session_state["page"] == "home":

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-section">
        <div class="hero-label">LaptiQ</div>
        <div class="hero-heading">
            <div class="hero-line-1">Know what your</div>
            <div class="hero-line-2">Laptop is Worth</div>
        </div>
        <div class="hero-subtitle">
            Enter your specs. Get a market price range. Understand why.
        </div>
        <div class="hero-accent">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Laptop Configuration Form ---
    st.markdown("""<div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg></div><div class="fs-title">LAPTOP CONFIGURATION</div></div>""", unsafe_allow_html=True)

    # Row 1: Brand · Processor · Clock Speed · Graphic Processor · Graphics Memory
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        brand = st.selectbox("Brand", [
            "Lenovo", "Asus", "HP", "MSI", "Apple", "Acer", "Dell", "Samsung", "Other"
        ])
    with c2:
        processor = st.text_input(
            "Processor",
            placeholder="e.g. Intel Core i7-13700H",
            help="Full processor name — e.g. AMD Ryzen 5 7530U, Apple M3 Pro, Snapdragon X Elite"
        )
    with c3:
        clock_speed = st.number_input("Clock Speed (GHz)", min_value=1.0, max_value=6.0, value=3.5, step=0.1, format="%.1f")
    with c4:
        graphic_processor = st.text_input(
            "Graphic Processor",
            placeholder="e.g. NVIDIA RTX 4060",
            help="e.g. NVIDIA RTX 4060, Intel Iris Xe, AMD Radeon 780M, Apple M3 GPU"
        )
    with c5:
        graphics_memory = st.selectbox(
            "Graphics Memory (GB)",
            [0, 2, 4, 6, 8, 12, 16, 24],
            format_func=lambda x: "Shared / Integrated" if x == 0 else f"{x} GB",
            help="Pick 'Shared / Integrated' for integrated graphics with no dedicated VRAM"
        )

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

    # Row 2: RAM · RAM Type · RAM Speed
    c1, c2, c3 = st.columns(3)
    with c1:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 24, 32, 48, 64, 128])
    with c2:
        ram_type = st.selectbox("RAM Type", ["DDR5", "DDR4", "LPDDR5", "LPDDR5X", "LPDDR4X", "Other"])
    with c3:
        ram_speed = st.number_input("RAM Speed (MHz)", min_value=1600, max_value=8000, value=4800, step=100)

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

    # Row 3: SSD · HDD
    c1, c2 = st.columns(2)
    with c1:
        ssd = st.selectbox("SSD Capacity", list(STORAGE_MAP.keys()) + ["None"], help="Select 'None' if no SSD")
    with c2:
        hdd = st.selectbox("HDD Capacity", ["None"] + list(STORAGE_MAP.keys()), help="Select 'None' if no HDD")

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

    # Row 4: Display Type · Touchscreen · Refresh Rate
    c1, c2, c3 = st.columns(3)
    with c1:
        display_type = st.selectbox("Display Type", [
            "LED", "IPS", "OLED", "AMOLED", "Mini-LED", "TN", "VA", "Other"
        ])
    with c2:
        display_touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    with c3:
        refresh_rate = st.selectbox(
            "Refresh Rate (Hz)",
            [60, 90, 120, 144, 165, 240, 360],
            format_func=lambda x: f"{x} Hz"
        )

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

    # Row 5: Resolution · Screen Size · Operating System · Weight
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        resolution = st.selectbox("Resolution", [
            "1366x768", "1920x1080", "1920x1200", "2560x1440",
            "2560x1600", "2880x1800", "3024x1964",
            "3456x2160", "3840x2160"
        ])
    with c2:
        screen_size = st.selectbox(
            "Screen Size",
            [11.6, 13.3, 13.6, 14.0, 15.6, 16.0, 17.3],
            index=4,
            format_func=lambda x: f'{x}"'
        )
    with c3:
        operating_system = st.selectbox("Operating System", ["Windows", "macOS", "ChromeOS", "Other"])
    with c4:
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1, format="%.1f")

    # PPI is auto-calculated — the model expects PPI but users find resolution + screen size more intuitive
    pixel_density = calculate_ppi(resolution, screen_size)

    # --- Predict Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)


    # === Prediction ==============================

    if predict_clicked:
        if not processor.strip():
            st.warning("Please enter a Processor — e.g. Intel Core i7-13700H")
        elif not graphic_processor.strip():
            st.warning("Please enter a Graphic Processor — e.g. NVIDIA RTX 4060, Intel Iris Xe")
        else:
            placeholder = st.empty()
            placeholder.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;padding:1.5rem;">
                <div class="spin-dot"></div>
                <div style="color:#7c7c90;font-size:0.9rem;font-weight:500;">
                    Crunching your specs…
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.8)

            ssd_val = 0.0 if ssd == "None" else float(STORAGE_MAP[ssd])
            hdd_val = 0.0 if hdd == "None" else float(STORAGE_MAP[hdd])

            input_data = {
                "Weight (Kg)":           float(weight),
                "Pixel Density (PPI)":   float(pixel_density),
                "Clock Speed (GHz)":     float(clock_speed),
                "RAM Speed (MHz)":       float(ram_speed),
                "Refresh Rate (Hz)":     float(refresh_rate),
                "Graphics Memory (GB)":  float(graphics_memory),
                "SSD Capacity (GB)":     ssd_val,
                "HDD Capacity (GB)":     hdd_val,
                "RAM (GB)":              float(ram),
                "Display Type":          display_type,
                "Display Touchscreen":   display_touchscreen,
                "RAM Type":              ram_type,
                "Brand":                 brand,
                "Operating System":      operating_system,
                "Processor":             processor.strip(),
                "Graphic Processor":     graphic_processor.strip(),
            }

            try:
                input_df   = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                price      = int(round(prediction[0]))

                margin      = price * 0.15
                lower_price = int(round((price - margin) / 500) * 500)
                upper_price = int(round((price + margin) / 500) * 500)

                formatted       = indian_format(price)
                formatted_lower = indian_format(lower_price)
                formatted_upper = indian_format(upper_price)

                placeholder.empty()
                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-box">
                        <div class="result-label">Estimated Price Range</div>
                        <div class="result-price">₹{formatted_lower} – ₹{formatted_upper}</div>
                        <div class="result-subprice">(Most likely around ₹{formatted})</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                placeholder.empty()
                st.error(f"Prediction failed: {e}")

    st.markdown("""
    <div class="footer-note">
        Predictions are based on training data and current market trends. Actual prices may vary.
    </div>
    """, unsafe_allow_html=True)


# === PAGE: MODEL INFO ==============================

elif st.session_state["page"] == "model_info":

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-section">
        <div class="hero-heading">
            <div class="hero-line-1">About the</div>
            <div class="hero-line-2">Model Behind It</div>
        </div>
        <div class="hero-subtitle">
            Lapti_Q estimates market value from specs — powered by a tuned XGBoost pipeline
            trained on ~8000+ laptops from the Indian market.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Model Stats ---
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
        </div>
        <div class="section-title">Model Performance</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">R² Score</div>
                <div class="sb-stat-icon blue">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                </div>
            </div>
            <div class="sb-stat-val-new blue">86.26%</div>
            <div class="sb-stat-sub">Higher is better</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Avg Prediction Error</div>
                <div class="sb-stat-icon orange">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                </div>
            </div>
            <div class="sb-stat-val-new orange">14.59%</div>
            <div class="sb-stat-sub">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Laptops Trained On</div>
                <div class="sb-stat-icon purple">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
                </div>
            </div>
            <div class="sb-stat-val-new purple">~8000+</div>
            <div class="sb-stat-sub">Indian market data</div>
        </div>
        """, unsafe_allow_html=True)

    # --- How It Works ---
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        </div>
        <div class="section-title">How It Works</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Enter your laptop specs", "Provide configuration details like CPU, RAM, GPU, storage, and more."),
        ("Lapti_Q engineers your input", "It cleans, process, and transform your specs for optimal prediction."),
        ("Get accurate market value", "Our tuned XGBoost model predicts the fair market price instantly."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="mi-step-card">
            <div class="mi-step-num">{i}</div>
            <div class="mi-step-content">
                <div class="mi-step-title">{title}</div>
                <div class="mi-step-desc">{desc}</div>
            </div>
            <div class="mi-step-chevron">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7c7c90" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        Built with XGBoost · sklearn Pipeline · Streamlit
    </div>
    """, unsafe_allow_html=True)
