"""
SKETCH2SHAPE AI - Modern UI/UX Redesign (Complete SaaS Edition)
TechZephyr 4.0 Hackathon • Indian Institute of Technology (IIT) Bhubaneswar
Strictly adheres to the approved reference design system:
- Pure white cards (#FFFFFF)
- Soft borders (#E2E8F0)
- Light slate page background (#F8FAFC)
- Dark navy sidebar (#0B132B) with vibrant blue active pill (#1D4ED8)
- Subtle blue accents (#2563EB, #EFF6FF)
- Professional typography and consistent rounded components
- Preserves all TripoSR cloud GPU backend, OpenCV preprocessing, and 3D export functionality.
"""

import os
import sys
import io
import time
from PIL import Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Configure page layout and branding
st.set_page_config(
    page_title="Sketch2Shape AI | From 2D Sketch to 3D Reality",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib
import modules.preprocessing
import modules.triposr_client
try:
    importlib.reload(modules.preprocessing)
    importlib.reload(modules.triposr_client)
except Exception:
    pass

from modules.preprocessing import preprocess_sketch, sketch_to_3d_obj, get_sketch_metrics
from modules.viewer3d import render_3d_viewer
from modules.creations import save_creation, get_all_creations, delete_creation
from modules.triposr_client import check_backend_health, reconstruct_triposr, normalize_url, is_valid_backend_url

# Ensure runtime directories exist
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_sketches")
COLAB_DIR = os.path.join(BASE_DIR, "colab")

for d in [OUTPUT_DIR, PREPROCESS_DIR, MODELS_DIR, ASSETS_DIR, SAMPLES_DIR, COLAB_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

if "prev_nav_page" not in st.session_state:
    st.session_state.prev_nav_page = "Home"

if "input_tab" not in st.session_state:
    st.session_state.input_tab = "Upload"

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "canvas_tool" not in st.session_state:
    st.session_state.canvas_tool = "Pen"

if "brush_size" not in st.session_state:
    st.session_state.brush_size = 4

if "brush_color" not in st.session_state:
    st.session_state.brush_color = "#000000"

if "selected_sketch" not in st.session_state:
    st.session_state.selected_sketch = None

if "sketch_source_name" not in st.session_state:
    st.session_state.sketch_source_name = None

if "processed_results" not in st.session_state:
    st.session_state.processed_results = None

if "sketch_metrics" not in st.session_state:
    st.session_state.sketch_metrics = None

if "current_model_path" not in st.session_state:
    st.session_state.current_model_path = None

if "current_model_format" not in st.session_state:
    st.session_state.current_model_format = "none"

if "current_engine" not in st.session_state:
    st.session_state.current_engine = "Awaiting Reconstruction"

if "object_prompt" not in st.session_state:
    st.session_state.object_prompt = ""

if "obj_type" not in st.session_state:
    st.session_state.obj_type = "📦 Furniture"

if "material" not in st.session_state:
    st.session_state.material = "🪵 Wood"

if "style" not in st.session_state:
    st.session_state.style = "🎨 Modern"

def get_configured_backend_url() -> str:
    """Read backend URL from environment variables, Streamlit secrets, or empty default."""
    env_url = os.environ.get("TRIPOSR_BACKEND_URL", "").strip()
    if env_url:
        return env_url
    try:
        if hasattr(st, "secrets"):
            if "TRIPOSR_BACKEND_URL" in st.secrets:
                return str(st.secrets["TRIPOSR_BACKEND_URL"]).strip()
            if "triposr" in st.secrets and isinstance(st.secrets["triposr"], dict):
                return str(st.secrets["triposr"].get("backend_url", "")).strip()
    except Exception:
        pass
    return ""

# AI Backend configuration
if "reconstruction_mode" not in st.session_state:
    st.session_state.reconstruction_mode = "TripoSR Neural AI (Cloud GPU)"

if "backend_url" not in st.session_state:
    st.session_state.backend_url = get_configured_backend_url()

if "backend_status" not in st.session_state:
    st.session_state.backend_status = {"online": False, "checked": False, "info": "Awaiting URL configuration"}


# --------------------------------------------------
# CONSISTENT DESIGN SYSTEM CSS
# --------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@500;600;700&display=swap');

/* Design System Variables */
:root {
    --bg-page: #F8FAFC;
    --card-bg: #FFFFFF;
    --sidebar-bg: #0B132B;
    --sidebar-active: #1D4ED8;
    --text-primary: #0F172A;
    --text-secondary: #64748B;
    --text-muted: #94A3B8;
    --border-color: #E2E8F0;
    --border-hover: #CBD5E1;
    --blue-accent: #2563EB;
    --blue-light: #EFF6FF;
    --btn-dark: #0A1128;
    --radius-card: 14px;
    --radius-elem: 8px;
    --radius-pill: 24px;
}

/* Global App Container */
body, .stApp {
    background-color: var(--bg-page) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: var(--text-primary) !important;
    overflow-x: hidden !important;
}

/* Ensure the main container expands to 100% available width */
.stAppViewContainer,
[data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
    padding-left: 0 !important;
    width: 100% !important;
}

.stMain,
[data-testid="stMain"],
[data-testid="stAppViewContainer"] > .stMain,
[data-testid="stAppViewContainer"] > section.main {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    padding-left: 0 !important;
    flex-grow: 1 !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1420px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Backdrop Overlay Behind Sidebar */
#sidebar-backdrop {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(11, 19, 43, 0.52) !important;
    backdrop-filter: blur(3px) !important;
    -webkit-backdrop-filter: blur(3px) !important;
    z-index: 999998 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

body.sidebar-open #sidebar-backdrop,
.sidebar-open #sidebar-backdrop {
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Floating Off-Canvas Sidebar Drawer */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;
    width: 295px !important;
    min-width: 295px !important;
    max-width: 295px !important;
    height: 100vh !important;
    z-index: 1000000 !important;
    background: #0B132B !important;
    border-right: 1px solid #1E293B !important;
    box-shadow: 8px 0 32px rgba(0, 0, 0, 0.45) !important;
    transform: translateX(-100%) !important;
    transition: transform 0.32s cubic-bezier(0.16, 1, 0.3, 1) !important;
    overflow-y: auto !important;
    display: block !important;
    visibility: visible !important;
    padding: 0 !important;
}

/* Open State: Slide in to Left Edge */
section[data-testid="stSidebar"].drawer-open,
body.sidebar-open section[data-testid="stSidebar"],
[data-testid="stSidebar"].drawer-open,
body.sidebar-open [data-testid="stSidebar"] {
    transform: translateX(0) !important;
}

/* Hide Default Streamlit Collapse & Expand Buttons */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Hamburger Button in Top Navbar */
.nav-hamburger-btn {
    background: transparent !important;
    border: none !important;
    padding: 6px !important;
    margin-right: 6px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    border-radius: 8px !important;
    color: #334155 !important;
    transition: all 0.15s ease !important;
}

.nav-hamburger-btn:hover {
    background-color: #F1F5F9 !important;
    color: #2563EB !important;
    transform: scale(1.05) !important;
}

.nav-hamburger-btn:active {
    transform: scale(0.95) !important;
}

/* Close Button Inside Sidebar Drawer */
.sidebar-close-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #94A3B8 !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.sidebar-close-btn:hover {
    background: rgba(255, 255, 255, 0.18) !important;
    color: #FFFFFF !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
    transform: scale(1.05) !important;
}

/* Sidebar Inner Content Styling */
section[data-testid="stSidebar"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebar"] > div {
    background: #0B132B !important;
    background-color: #0B132B !important;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Sidebar Expander Styling */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #101B38 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    margin: 8px 12px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #60A5FA !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #0B132B !important;
    border-top: 1px solid #1E293B !important;
}

.sidebar-brand-box {
    padding: 24px 20px 16px 20px;
    text-align: center;
}

.sidebar-logo-icon {
    width: 44px;
    height: 44px;
    margin: 0 auto 12px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.sidebar-brand-title {
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #FFFFFF !important;
    margin: 0;
}

.sidebar-brand-subtitle {
    font-size: 11px;
    color: #94A3B8 !important;
    margin: 4px 0 0 0;
}

/* Sidebar Navigation Radio Menu */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
    padding: 0 12px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background-color: transparent !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin: 0 !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    border: none !important;
}

/* Hide Streamlit radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #CBD5E1 !important;
    margin: 0 !important;
    transition: color 0.15s ease !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background-color: #1E293B !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover p {
    color: #FFFFFF !important;
}

/* Active navigation item pill */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #1D4ED8 !important;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.45) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

.sidebar-footer-box {
    margin-top: 24px;
    padding: 16px 18px 0 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar-event-title {
    font-size: 13px;
    font-weight: 700;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-event-inst {
    font-size: 11px;
    color: #94A3B8 !important;
    margin-top: 3px;
    padding-left: 24px;
}

.sidebar-event-tag {
    font-size: 10px;
    color: #64748B !important;
    margin-top: 2px;
    padding-left: 24px;
    font-style: italic;
}

.sidebar-wave {
    margin-top: 16px;
    opacity: 0.85;
}

/* Top Navbar - Mathematically Centered Calligraphy Heading */
.top-navbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 8px 20px;
    display: grid !important;
    grid-template-columns: 1fr auto 1fr !important;
    align-items: center !important;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    min-height: 64px;
    position: relative;
}

.top-nav-left {
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.top-nav-center {
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.top-nav-title-script {
    font-family: 'Dancing Script', 'Caveat', 'Segoe Script', 'Brush Script MT', cursive !important;
    font-size: 38px !important;
    font-weight: 600 !important;
    color: #0B132B !important;
    letter-spacing: 0.02em !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.15 !important;
    user-select: text !important;
    white-space: nowrap !important;
}

.top-nav-right-spacer {
    display: block;
    width: 100%;
    pointer-events: none;
}

@media (max-width: 992px) {
    .top-nav-title-script {
        font-size: 32px !important;
    }
    .top-navbar {
        min-height: 56px;
        padding: 6px 16px;
    }
}

@media (max-width: 640px) {
    .top-nav-title-script {
        font-size: 26px !important;
    }
    .top-navbar {
        min-height: 50px;
        padding: 6px 12px;
    }
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 100%);
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.hero-text h2 {
    font-size: 22px;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
    line-height: 1.25;
}

.hero-text p {
    font-size: 13px;
    color: #64748B;
    margin: 0;
}

/* Consistent Card Styling Across All Pages */
.saas-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.card-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.card-title-text {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
}

/* Segmented Control for Upload / Draw Switch */
div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] {
    background: #F1F5F9 !important;
    padding: 3px !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    gap: 4px !important;
    border: 1px solid #E2E8F0 !important;
    justify-content: flex-end !important;
}

div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 5px 14px !important;
    border-radius: 6px !important;
    margin: 0 !important;
    cursor: pointer !important;
    border: none !important;
}

div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] > label p {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    margin: 0 !important;
}

div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] > label:has(input:checked) {
    background-color: #0A1128 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

div[data-testid="stRadio"]:has(input[aria-label="📤 Upload"]) div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF !important;
}

/* Upload Dropzone Styling */
.dropzone-container {
    border: 2px dashed #93C5FD;
    border-radius: 12px;
    padding: 24px 16px 16px 16px;
    text-align: center;
    background: #FAFCFF;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.dropzone-container:hover {
    border-color: #2563EB;
    background: #EFF6FF;
}

.dropzone-title {
    font-size: 14px;
    font-weight: 700;
    color: #0F172A;
    margin: 8px 0 2px 0;
}

.dropzone-or {
    font-size: 11px;
    color: #94A3B8;
    margin-bottom: 4px;
}

/* Native File Uploader Styling */
[data-testid="stFileUploader"] {
    background: transparent !important;
    padding: 0 !important;
    margin-top: -6px !important;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    text-align: center !important;
}

[data-testid="stFileUploader"] button {
    background-color: #0A1128 !important;
    color: #FFFFFF !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 20px !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(10, 17, 40, 0.2) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #172554 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}

/* Sample divider */
.sample-divider {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 14px 0 12px 0;
}

.sample-divider::before, .sample-divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #E2E8F0;
}

.sample-divider span {
    padding: 0 10px;
    font-size: 11px;
    color: #64748B;
    font-weight: 500;
}

/* Bottom How It Works Workflow Bar */
.bottom-workflow-bar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    margin-top: 8px;
}

.workflow-steps-group {
    display: flex;
    align-items: center;
    gap: 14px;
    flex: 1;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.step-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #EFF6FF;
    color: #2563EB;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.step-text-title {
    font-size: 12px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}

.step-text-desc {
    font-size: 10px;
    color: #64748B;
    margin: 0;
}

.step-arrow {
    color: #CBD5E1;
    font-size: 14px;
}

/* Text Inputs, Selectboxes & Sliders */
.stTextInput > div > div > input {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
    font-size: 13px !important;
}

/* Buttons */
.stButton > button {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background-color: #F8FAFC !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
}

/* Primary Action Dark Pill Buttons */
.stButton > button[kind="primary"] {
    background-color: #0A1128 !important;
    border: 1px solid #0A1128 !important;
    border-radius: 24px !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 12px rgba(10, 17, 40, 0.2) !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #172554 !important;
    border-color: #172554 !important;
    transform: translateY(-1px) !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
}

.stDownloadButton > button:hover {
    background-color: #F8FAFC !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
}

/* Alerts */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div style="display:flex; justify-content:flex-end; margin-bottom:-8px;">
            <button id="sidebar-close-btn" class="sidebar-close-btn" aria-label="Close Sidebar" title="Close Navigation" onclick="
                var b = document.body;
                var s = document.querySelector('[data-testid=&quot;stSidebar&quot;]');
                b.classList.remove('sidebar-open');
                if (s) s.classList.remove('drawer-open');
            ">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
        <div class="sidebar-logo-icon">
            <svg width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <polygon points="24,4 44,15 24,26 4,15" fill="#3B82F6" stroke="#60A5FA" stroke-width="1.5" stroke-linejoin="round"/>
                <polygon points="4,15 24,26 24,44 4,33" fill="#1D4ED8" stroke="#3B82F6" stroke-width="1.5" stroke-linejoin="round"/>
                <polygon points="24,26 44,15 44,33 24,44" fill="#1E40AF" stroke="#2563EB" stroke-width="1.5" stroke-linejoin="round"/>
                <circle cx="24" cy="24" r="3" fill="#FFFFFF"/>
            </svg>
        </div>
        <h2 class="sidebar-brand-title">SKETCH2SHAPE AI</h2>
        <p class="sidebar-brand-subtitle">From 2D Sketch to 3D Reality</p>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Home",
        "Sketch Studio",
        "My Creations",
        "How It Works",
        "About Project"
    ]

    nav_icons = {
        "Home": "🏠  Home",
        "Sketch Studio": "✏️  Sketch Studio",
        "My Creations": "📦  My Creations",
        "How It Works": "📖  How It Works",
        "About Project": "ℹ️  About Project"
    }

    curr_idx = 0
    for idx, opt in enumerate(nav_options):
        if st.session_state.nav_page == opt:
            curr_idx = idx
            break

    selected_nav = st.radio(
        "Navigation Menu",
        nav_options,
        index=curr_idx,
        format_func=lambda x: nav_icons.get(x, x),
        label_visibility="collapsed",
        key="main_nav_radio"
    )
    if selected_nav != st.session_state.prev_nav_page:
        st.session_state.prev_nav_page = selected_nav
        st.session_state.nav_page = selected_nav
        st.session_state.sidebar_open = False
    else:
        st.session_state.nav_page = selected_nav

    # Cloud AI Configuration Drawer
    with st.expander("⚡ Cloud AI Settings", expanded=False):
        st.markdown("<div style='font-size:12px; color:#CBD5E1; margin-bottom:8px; font-weight:600;'>Neural Engine Setup (Google Colab Tesla T4)</div>", unsafe_allow_html=True)

        backend_choice = st.radio(
            "Inference Backend",
            ["TripoSR (Cloud GPU)", "Local 3D Engine (Offline Mode)"],
            index=0 if "TripoSR" in st.session_state.reconstruction_mode else 1,
            label_visibility="collapsed",
            key="side_backend_select"
        )
        st.session_state.reconstruction_mode = backend_choice

        if "TripoSR" in backend_choice:
            st.markdown("""
            <div style="font-size:11px; color:#94A3B8; margin-bottom:8px; line-height:1.4;">
                1. Run <code style="color:#60A5FA;">colab/server.py</code> on Google Colab (Tesla T4 GPU)<br>
                2. Copy the active Cloudflare Tunnel URL<br>
                3. Paste below to connect:
            </div>
            """, unsafe_allow_html=True)

            url_val = st.text_input(
                "Colab Backend URL",
                value=st.session_state.backend_url,
                placeholder="https://xxxx.trycloudflare.com",
                key="side_url_input",
                help="Paste the active public .trycloudflare.com tunnel URL from your Colab notebook."
            )
            st.session_state.backend_url = url_val.strip()

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                if st.button("🔌 Ping Backend", use_container_width=True, key="side_ping_btn"):
                    with st.spinner("Connecting..."):
                        h = check_backend_health(st.session_state.backend_url, timeout=4)
                        st.session_state.backend_status = h
            with c_btn2:
                if st.button("🗑️ Clear URL", use_container_width=True, key="side_clear_url_btn"):
                    st.session_state.backend_url = ""
                    st.session_state.backend_status = {"online": False, "checked": False, "info": "Cleared"}
                    st.rerun()

            status_info = st.session_state.backend_status
            is_val, val_msg = is_valid_backend_url(st.session_state.backend_url)

            if not is_val:
                st.markdown(f"""
                <div style="background:#1E293B; border:1px solid #475569; border-radius:6px; padding:6px 8px; margin-top:6px; font-size:10px; color:#94A3B8;">
                    ⚪ Not Configured: {val_msg}
                </div>
                """, unsafe_allow_html=True)
            elif status_info.get("online"):
                st.markdown(f"""
                <div style="background:#064E3B; border:1px solid #059669; border-radius:6px; padding:6px 8px; margin-top:6px; font-size:11px;">
                    <div style="color:#34D399; font-weight:700;">🟢 Online: {status_info.get('model', 'TripoSR')}</div>
                    <div style="color:#A7F3D0; font-size:10px;">{status_info.get('gpu_name', 'Tesla T4')} ({status_info.get('latency_ms', 0)}ms)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                err = status_info.get("error") or "Offline"
                st.markdown(f"""
                <div style="background:#451A1A; border:1px solid #DC2626; border-radius:6px; padding:6px 8px; margin-top:6px; font-size:10px; color:#FCA5A5;">
                    🔴 Offline: {err[:60]}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1E293B; border:1px solid #3B82F6; border-radius:6px; padding:8px; margin-top:6px; font-size:11px; color:#93C5FD;">
                ⚙️ <strong>Offline Mode Active:</strong> 3D geometry is generated locally via OpenCV contour analysis & structural synthesis. No cloud GPU connection required.
            </div>
            """, unsafe_allow_html=True)

    # Sidebar Bottom Event Branding & Wave Decoration
    st.markdown("""
    <div class="sidebar-footer-box">
        <div class="sidebar-event-title">
            <span style="font-size:16px;">🏆</span>
            <span>TechZephyr 4.0</span>
        </div>
        <div class="sidebar-event-inst">IIT Bhubaneswar</div>
        <div class="sidebar-event-tag">AI-Powered 3D Reconstruction</div>
    </div>
    <div class="sidebar-wave">
        <svg viewBox="0 0 280 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%; display:block;">
            <path d="M0 20 C80 5, 160 38, 280 12 L280 48 L0 48 Z" fill="#172554" fill-opacity="0.6"/>
            <path d="M0 28 C90 14, 180 42, 280 25 L280 48 L0 48 Z" fill="#1D4ED8" fill-opacity="0.45"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# BACKDROP OVERLAY & TOP NAVBAR
# --------------------------------------------------
st.markdown("""
<div id="sidebar-backdrop" class="sidebar-backdrop" onclick="
    var b = document.body;
    var s = document.querySelector('[data-testid=&quot;stSidebar&quot;]');
    b.classList.remove('sidebar-open');
    if (s) s.classList.remove('drawer-open');
"></div>

<div class="top-navbar">
    <div class="top-nav-left">
        <button id="hamburger-menu-btn" class="nav-hamburger-btn" aria-label="Toggle Navigation Menu" title="Toggle Navigation Menu" onclick="
            var b = document.body;
            var s = document.querySelector('[data-testid=&quot;stSidebar&quot;]');
            if (b.classList.contains('sidebar-open')) {
                b.classList.remove('sidebar-open');
                if (s) s.classList.remove('drawer-open');
            } else {
                b.classList.add('sidebar-open');
                if (s) s.classList.add('drawer-open');
            }
        ">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
        </button>
    </div>
    <div class="top-nav-center">
        <h1 class="top-nav-title-script">Sketch2Shape AI</h1>
    </div>
    <div class="top-nav-right-spacer"></div>
</div>
""", unsafe_allow_html=True)

# Companion Client-Side Controller for Seamless Drawer Transitions
components.html("""
<script>
(function() {
    function setupDrawer() {
        try {
            const doc = window.parent.document;
            if (!doc) return;

            const b = doc.body;
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            const hamburger = doc.getElementById('hamburger-menu-btn');
            const backdrop = doc.getElementById('sidebar-backdrop');
            const closeBtn = doc.getElementById('sidebar-close-btn');

            function openD() {
                b.classList.add('sidebar-open');
                if (sidebar) sidebar.classList.add('drawer-open');
            }
            function closeD() {
                b.classList.remove('sidebar-open');
                if (sidebar) sidebar.classList.remove('drawer-open');
            }
            function toggleD(e) {
                if (e) { e.preventDefault(); e.stopPropagation(); }
                if (b.classList.contains('sidebar-open')) closeD();
                else openD();
            }

            if (hamburger && !hamburger._boundDrawer) {
                hamburger._boundDrawer = true;
                hamburger.addEventListener('click', toggleD);
            }
            if (backdrop && !backdrop._boundDrawer) {
                backdrop._boundDrawer = true;
                backdrop.addEventListener('click', function(e) {
                    e.preventDefault();
                    closeD();
                });
            }
            if (closeBtn && !closeBtn._boundDrawer) {
                closeBtn._boundDrawer = true;
                closeBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    closeD();
                });
            }

            const radioItems = doc.querySelectorAll('[data-testid="stSidebar"] div[role="radiogroup"] label');
            radioItems.forEach(function(item) {
                if (!item._boundDrawerNav) {
                    item._boundDrawerNav = true;
                    item.addEventListener('click', function() {
                        setTimeout(closeD, 120);
                    });
                }
            });

            if (!doc._boundDrawerEscape) {
                doc._boundDrawerEscape = true;
                doc.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') closeD();
                });
            }
        } catch (err) {
            console.error('Drawer script error:', err);
        }
    }

    setupDrawer();
    setTimeout(setupDrawer, 150);
    setTimeout(setupDrawer, 400);
    setTimeout(setupDrawer, 800);
})();
</script>
""", height=0, width=0)


# ==================================================
# SHARED FUNCTION: RUN 3D RECONSTRUCTION
# ==================================================
def execute_reconstruction():
    if st.session_state.selected_sketch is None:
        st.warning("⚠️ Please provide a sketch by uploading an image, drawing on canvas, or picking a sample first.")
        return

    input_sketch_path = os.path.join(OUTPUT_DIR, "input_sketch.png")
    st.session_state.selected_sketch.save(input_sketch_path)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Stage 1: OpenCV Preprocessing
    status_text.markdown("**Stage 1/4: Analyzing sketch & running OpenCV noise filtration...**")
    progress_bar.progress(25)
    results = preprocess_sketch(input_sketch_path, PREPROCESS_DIR)
    st.session_state.processed_results = results
    st.session_state.sketch_metrics = get_sketch_metrics(input_sketch_path)

    ai_success = False
    target_glb = os.path.join(MODELS_DIR, "reconstructed_triposr.glb")
    target_obj = os.path.join(MODELS_DIR, "reconstructed_mesh.obj")

    # Stage 2 & 3: TripoSR Cloud Backend
    if "TripoSR" in st.session_state.reconstruction_mode:
        is_valid_url, val_msg = is_valid_backend_url(st.session_state.backend_url)
        if not is_valid_url:
            st.warning(f"⚠️ TripoSR Cloud Backend Not Configured: {val_msg}")
            st.info("ℹ️ To use Cloud Neural AI (Tesla T4 GPU): Run the Colab notebook (`colab/server.py`), copy the public `.trycloudflare.com` URL, and paste it into '⚡ Cloud AI Settings' in the sidebar.")
            st.info("🔄 Engaging Local 3D Structural Engine (Offline Mode)...")
        else:
            status_text.markdown(f"**Stage 2/4: Transmitting sketch to TripoSR Cloud Backend ({st.session_state.backend_url})...**")
            progress_bar.progress(50)

            status_text.markdown("**Stage 3/4: Marching Cubes neural field reconstruction on Tesla T4 GPU...**")
            progress_bar.progress(75)

            tripo_res = reconstruct_triposr(
                backend_url=st.session_state.backend_url,
                image_path=input_sketch_path,
                output_glb_path=target_glb,
                output_obj_path=target_obj,
                mc_resolution=256
            )

            if tripo_res.get("success"):
                ai_success = True
                status_text.markdown("**Stage 4/4: Compiling WebGL 3D scene & PBR textures...**")
                progress_bar.progress(100)
                time.sleep(0.2)

                st.session_state.current_model_path = target_glb
                st.session_state.current_model_format = "glb"
                st.session_state.current_engine = "TripoSR Neural AI (Cloud Tesla T4)"

                clean_type = st.session_state.obj_type.replace("📦 ", "").replace("🚗 ", "").replace("🏠 ", "").replace("🏛️ ", "").replace("📱 ", "").replace("✨ ", "")
                clean_mat = st.session_state.material.replace("🪵 ", "").replace("⚙️ ", "").replace("🧱 ", "").replace("🪟 ", "").replace("🏺 ", "").replace("🪄 ", "")
                clean_style = st.session_state.style.replace("🎨 ", "").replace("📸 ", "").replace("📐 ", "").replace("🧸 ", "").replace("🏭 ", "")

                save_creation(
                    sketch_image=st.session_state.selected_sketch,
                    processed_paths=results,
                    obj_path=target_obj if os.path.exists(target_obj) else None,
                    glb_path=target_glb,
                    prompt=st.session_state.object_prompt,
                    obj_type=clean_type,
                    material=clean_mat,
                    style=clean_style,
                    engine="TripoSR Neural AI",
                    vertex_count=tripo_res.get("vertex_count", 0),
                    face_count=tripo_res.get("face_count", 0)
                )

                status_text.empty()
                progress_bar.empty()
                st.success(f"✓ TripoSR Neural 3D Reconstruction succeeded in {tripo_res.get('inference_time_sec')}s on {tripo_res.get('gpu_name', 'Tesla T4')}!")
                st.rerun()
            else:
                err_msg = tripo_res.get("error", "Unknown error")
                st.warning(f"⚠️ Cloud TripoSR Backend Unreachable: {err_msg}")
                st.info("🔄 Engaging Local 3D Structural Engine as labeled fallback...")

    # Local Structural 3D Reconstruction Engine (Offline Mode)
    if not ai_success:
        status_text.markdown("**Stage 3/4: Synthesizing 3D object topology & manifold surface geometry...**")
        progress_bar.progress(75)
        time.sleep(0.2)

        success_fallback = sketch_to_3d_obj(
            binary_img_path=input_sketch_path,
            output_obj_path=target_obj,
            depth=None,
            output_glb_path=target_glb,
            prompt=st.session_state.object_prompt,
            obj_type=st.session_state.obj_type
        )
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        if success_fallback and (os.path.exists(target_obj) or os.path.exists(target_glb)):
            chosen_model = target_glb if (target_glb and os.path.exists(target_glb)) else target_obj
            st.session_state.current_model_path = chosen_model
            st.session_state.current_model_format = "glb" if chosen_model.endswith(".glb") else "obj"
            st.session_state.current_engine = "Local 3D Structural Engine (Offline Mode)"

            clean_type = st.session_state.obj_type.replace("📦 ", "").replace("🚗 ", "").replace("🏠 ", "").replace("🏛️ ", "").replace("📱 ", "").replace("✨ ", "")
            clean_mat = st.session_state.material.replace("🪵 ", "").replace("⚙️ ", "").replace("🧱 ", "").replace("🪟 ", "").replace("🏺 ", "").replace("🪄 ", "")
            clean_style = st.session_state.style.replace("🎨 ", "").replace("📸 ", "").replace("📐 ", "").replace("🧸 ", "").replace("🏭 ", "")

            save_creation(
                sketch_image=st.session_state.selected_sketch,
                processed_paths=results,
                obj_path=target_obj if os.path.exists(target_obj) else None,
                glb_path=target_glb if os.path.exists(target_glb) else None,
                prompt=st.session_state.object_prompt,
                obj_type=clean_type,
                material=clean_mat,
                style=clean_style,
                engine="Local 3D Structural Engine (Offline Mode)"
            )

            st.success("✓ 3D Asset Reconstructed Successfully via Local 3D Structural Engine (Offline Mode).")
            st.rerun()
        else:
            st.error("Failed to generate 3D geometry from sketch contours.")


# ==================================================
# VIEW: HOME
# ==================================================
if st.session_state.nav_page == "Home":

    # HERO SECTION
    st.markdown("""
    <div class="hero-box">
        <div class="hero-text">
            <h2>Transform your hand-drawn ideas<br>into intelligent 3D digital assets.</h2>
            <p>Sketch it. Generate it. Explore it.</p>
        </div>
        <div class="hero-visual">
            <svg width="280" height="96" viewBox="0 0 280 96" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="chairVignette" cx="75%" cy="50%" r="45%">
                        <stop offset="0%" stop-color="#DBEAFE" stop-opacity="0.95"/>
                        <stop offset="100%" stop-color="#EFF6FF" stop-opacity="0"/>
                    </radialGradient>
                </defs>
                <g transform="translate(10, 4)">
                    <path d="M22 10 C22 5, 48 5, 48 10 L46 48 C46 50, 24 50, 24 48 Z" stroke="#334155" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M29 16 L29 46" stroke="#94A3B8" stroke-width="1.2" stroke-linecap="round"/>
                    <path d="M41 16 L41 46" stroke="#94A3B8" stroke-width="1.2" stroke-linecap="round"/>
                    <path d="M18 48 L52 48 L59 57 L11 57 Z" stroke="#1E293B" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <line x1="14" y1="57" x2="12" y2="84" stroke="#1E293B" stroke-width="1.8" stroke-linecap="round"/>
                    <line x1="56" y1="57" x2="58" y2="84" stroke="#1E293B" stroke-width="1.8" stroke-linecap="round"/>
                    <line x1="24" y1="57" x2="25" y2="78" stroke="#64748B" stroke-width="1.4" stroke-linecap="round"/>
                    <line x1="46" y1="57" x2="45" y2="78" stroke="#64748B" stroke-width="1.4" stroke-linecap="round"/>
                </g>
                <g transform="translate(110, 42)">
                    <line x1="0" y1="6" x2="24" y2="6" stroke="#64748B" stroke-width="2" stroke-linecap="round"/>
                    <polyline points="18,1 24,6 18,11" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </g>
                <g transform="translate(160, 2)">
                    <ellipse cx="50" cy="46" rx="48" ry="42" fill="url(#chairVignette)"/>
                    <ellipse cx="50" cy="85" rx="30" ry="5" fill="#94A3B8" fill-opacity="0.35"/>
                    <path d="M32 14 C32 8, 68 8, 68 14 L66 49 C66 52, 34 52, 34 49 Z" fill="#E2E8F0"/>
                    <path d="M35 15 C35 10, 65 10, 65 15 L63 48 C63 50, 37 50, 37 48 Z" fill="#F8FAFC"/>
                    <path d="M66 14 L69 16 L67 49 L64 48 Z" fill="#CBD5E1"/>
                    <path d="M28 50 C28 48, 72 48, 72 50 L77 60 C77 62, 23 62, 23 60 Z" fill="#E2E8F0"/>
                    <path d="M29 51 C29 49, 71 49, 71 51 L75 59 C75 61, 25 61, 25 59 Z" fill="#FFFFFF"/>
                    <path d="M23 60 L77 60 L75 64 L25 64 Z" fill="#94A3B8"/>
                    <line x1="28" y1="64" x2="25" y2="85" stroke="#334155" stroke-width="3" stroke-linecap="round"/>
                    <line x1="72" y1="64" x2="75" y2="85" stroke="#334155" stroke-width="3" stroke-linecap="round"/>
                    <line x1="38" y1="63" x2="39" y2="80" stroke="#64748B" stroke-width="2.2" stroke-linecap="round"/>
                    <line x1="62" y1="63" x2="61" y2="80" stroke="#64748B" stroke-width="2.2" stroke-linecap="round"/>
                </g>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TWO-COLUMN WORKSPACE
    col_left, col_right = st.columns([1.05, 0.95], gap="medium")

    # LEFT COLUMN
    with col_left:
        # Card 1: Input Your Sketch
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        
        # Header with title on left and segmented switch on right
        c_hdr1, c_hdr2 = st.columns([1.2, 0.8])
        with c_hdr1:
            st.markdown("""
            <div class="card-title-text" style="padding-top: 6px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"></path>
                </svg>
                <span>Input Your Sketch</span>
            </div>
            """, unsafe_allow_html=True)
        with c_hdr2:
            tab_choice = st.radio(
                "Input Mode",
                ["📤 Upload", "✏️ Draw"],
                horizontal=True,
                index=0 if st.session_state.input_tab == "Upload" else 1,
                label_visibility="collapsed",
                key="input_segmented_switch"
            )
            st.session_state.input_tab = "Upload" if "Upload" in tab_choice else "Draw"

        # OPTION A: UPLOAD
        if st.session_state.input_tab == "Upload":
            st.markdown("""
            <div class="dropzone-container">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
                    <path d="M12 12v9"></path>
                    <path d="m16 16-4-4-4 4"></path>
                </svg>
                <div class="dropzone-title">Drag & Drop your sketch here</div>
                <div class="dropzone-or">or</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Browse Files",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="dropzone_file_uploader"
            )

            if uploaded_file is not None:
                try:
                    up_img = Image.open(uploaded_file).convert("RGB")
                    st.session_state.selected_sketch = up_img
                    st.session_state.sketch_source_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error opening image: {e}")

            st.markdown("""
            <div style="text-align:center; font-size:11px; color:#64748B; margin-top:-2px; margin-bottom:12px;">
                Supported: PNG, JPG, JPEG &nbsp;|&nbsp; Max Size: 10MB
            </div>
            """, unsafe_allow_html=True)

        # OPTION B: DRAW
        else:
            st.caption("Draw your concept with your mouse or stylus on the digital canvas below.")
            
            c_tools1, c_tools2, c_tools3 = st.columns([1, 1, 1])
            with c_tools1:
                tool_mode = st.radio("Tool", ["Pen", "Eraser"], horizontal=True, key="canvas_pen_eraser", label_visibility="collapsed")
            with c_tools2:
                stroke_w = st.slider("Width", 1, 20, 4, key="canvas_stroke_slider", label_visibility="collapsed")
            with c_tools3:
                stroke_c = st.color_picker("Color", "#000000", key="canvas_color_picker", label_visibility="collapsed") if tool_mode == "Pen" else "#FFFFFF"

            from streamlit_drawable_canvas import st_canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_w,
                stroke_color=stroke_c,
                background_color="#FFFFFF",
                height=320,
                width=520,
                drawing_mode="freedraw",
                key=f"saas_canvas_{st.session_state.canvas_key}",
                return_image_data=True
            )

            if canvas_result.image_data is not None:
                arr = canvas_result.image_data.astype(np.uint8)
                if np.any(arr[:, :, :3] < 250):
                    drawn_img = Image.fromarray(arr, mode="RGBA").convert("RGB")
                    st.session_state.selected_sketch = drawn_img
                    st.session_state.sketch_source_name = "Digital Canvas Drawing"

            draw_c1, draw_c2 = st.columns(2)
            with draw_c1:
                if st.button("🗑️ Clear Canvas", use_container_width=True, key="clear_canvas_main"):
                    st.session_state.canvas_key += 1
                    st.session_state.selected_sketch = None
                    st.session_state.sketch_source_name = None
                    st.rerun()
            with draw_c2:
                if st.session_state.selected_sketch is not None:
                    buf = io.BytesIO()
                    st.session_state.selected_sketch.save(buf, format="PNG")
                    st.download_button("💾 Download Sketch", data=buf.getvalue(), file_name="sketch.png", mime="image/png", use_container_width=True)

        # SAMPLE SKETCHES ROW
        st.markdown("""
        <div class="sample-divider">
            <span>or try our sample sketches</span>
        </div>
        """, unsafe_allow_html=True)

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            if st.button("🪑 Chair", use_container_width=True, key="sample_btn_chair"):
                chair_path = os.path.join(SAMPLES_DIR, "chair_sketch.png")
                if os.path.exists(chair_path):
                    st.session_state.selected_sketch = Image.open(chair_path).convert("RGB")
                    st.session_state.sketch_source_name = "Sample Chair"
                    st.session_state.object_prompt = "Modern ergonomic dining chair with polished finish."
                    st.session_state.current_model_path = os.path.join(ASSETS_DIR, "sample_chair.obj")
                    st.session_state.current_model_format = "obj"
                    st.session_state.current_engine = "Curated Reference Asset"
                    st.rerun()

        with s_col2:
            if st.button("🏺 Vase", use_container_width=True, key="sample_btn_vase"):
                vase_path = os.path.join(SAMPLES_DIR, "vase_sketch.png")
                if os.path.exists(vase_path):
                    st.session_state.selected_sketch = Image.open(vase_path).convert("RGB")
                    st.session_state.sketch_source_name = "Sample Vase"
                    st.session_state.object_prompt = "Elegant decorative ceramic flower vase."
                    st.session_state.current_model_path = os.path.join(ASSETS_DIR, "sample_vase.obj")
                    st.session_state.current_model_format = "obj"
                    st.session_state.current_engine = "Curated Reference Asset"
                    st.rerun()

        with s_col3:
            if st.button("☕ Mug", use_container_width=True, key="sample_btn_mug"):
                mug_path = os.path.join(SAMPLES_DIR, "mug_sketch.png")
                if os.path.exists(mug_path):
                    st.session_state.selected_sketch = Image.open(mug_path).convert("RGB")
                    st.session_state.sketch_source_name = "Sample Mug"
                    st.session_state.object_prompt = "Cylindrical ceramic coffee mug with ergonomic handle."
                    st.session_state.current_model_path = os.path.join(ASSETS_DIR, "sample_mug.obj")
                    st.session_state.current_model_format = "obj"
                    st.session_state.current_engine = "Curated Reference Asset"
                    st.rerun()

        # Selected Sketch Details Row
        if st.session_state.selected_sketch is not None:
            st.markdown("<hr style='margin: 14px 0 10px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            p_c1, p_c2 = st.columns([1, 2.5])
            with p_c1:
                st.image(st.session_state.selected_sketch, width=105)
            with p_c2:
                name = st.session_state.sketch_source_name or "Sketch Image"
                w, h = st.session_state.selected_sketch.size
                st.markdown(f"**{name}**")
                st.caption(f"Dimensions: {w} × {h} px &nbsp;|&nbsp; Status: Ready")
                if st.button("🗑️ Remove Sketch", key="remove_active_sketch", use_container_width=False):
                    st.session_state.selected_sketch = None
                    st.session_state.sketch_source_name = None
                    st.session_state.processed_results = None
                    st.session_state.current_model_path = None
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Describe Your 3D Object (Optional)
        st.markdown("""
        <div class="saas-card" style="padding: 16px 20px;">
            <div class="card-header-row" style="margin-bottom: 10px;">
                <div class="card-title-text" style="font-size: 14px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 20h9"></path>
                        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                    </svg>
                    <span>Describe Your 3D Object (Optional)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        desc_c1, desc_c2, desc_c3, desc_c4 = st.columns([2.5, 1.25, 1.15, 1.15])
        with desc_c1:
            st.session_state.object_prompt = st.text_input(
                "Description",
                value=st.session_state.object_prompt,
                placeholder="e.g. A modern wooden chair with four legs...",
                label_visibility="collapsed"
            )
        with desc_c2:
            st.session_state.obj_type = st.selectbox(
                "Type",
                ["📦 Furniture", "🚗 Vehicle", "🏠 Household", "🏛️ Architecture", "📱 Electronics", "✨ Other"],
                index=0,
                label_visibility="collapsed"
            )
        with desc_c3:
            st.session_state.material = st.selectbox(
                "Material",
                ["🪵 Wood", "⚙️ Metal", "🧱 Plastic", "🪟 Glass", "🏺 Ceramic", "🪄 Custom"],
                index=0,
                label_visibility="collapsed"
            )
        with desc_c4:
            st.session_state.style = st.selectbox(
                "Style",
                ["🎨 Modern", "📸 Realistic", "📐 Minimalist", "🧸 Cartoon", "🏭 Industrial"],
                index=0,
                label_visibility="collapsed"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN: 3D MODEL PREVIEW
    with col_right:
        active_model = st.session_state.current_model_path

        # Render 3D Model Viewport (shows isometric standby cube or interactive Three.js)
        render_3d_viewer(
            model_file_path=active_model,
            height=495,
            model_title="3D Model Preview"
        )

        # Download Buttons
        if active_model and os.path.exists(active_model):
            model_base = os.path.basename(active_model).replace(".glb", "").replace(".obj", "").replace("_", " ").title()
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            d_c1, d_c2 = st.columns(2)
            
            with d_c1:
                glb_file = active_model if active_model.endswith(".glb") else active_model.replace(".obj", ".glb")
                if os.path.exists(glb_file):
                    with open(glb_file, "rb") as fg:
                        st.download_button(
                            "📥 Download GLB",
                            data=fg,
                            file_name=f"{model_base.lower().replace(' ', '_')}.glb",
                            mime="model/gltf-binary",
                            use_container_width=True
                        )
                else:
                    st.button("📦 GLB Export (Cloud Only)", disabled=True, use_container_width=True)

            with d_c2:
                obj_file = active_model if active_model.endswith(".obj") else active_model.replace(".glb", ".obj")
                if os.path.exists(obj_file):
                    with open(obj_file, "rb") as fo:
                        st.download_button(
                            "📥 Download Wavefront OBJ",
                            data=fo,
                            file_name=f"{model_base.lower().replace(' ', '_')}.obj",
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.button("📥 OBJ Export", disabled=True, use_container_width=True)

    # BOTTOM WORKFLOW BAR: HOW IT WORKS & GENERATE ACTION
    st.markdown("""
    <div class="bottom-workflow-bar">
        <div style="font-size:14px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:6px;">
            <span>▶</span>
            <span>How It Works</span>
        </div>
        <div class="workflow-steps-group">
            <div class="step-item">
                <div class="step-circle">1</div>
                <div>
                    <div class="step-text-title">Input Sketch</div>
                    <div class="step-text-desc">Upload or draw your concept</div>
                </div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-circle">2</div>
                <div>
                    <div class="step-text-title">AI Processing</div>
                    <div class="step-text-desc">Our AI reconstructs the 3D model</div>
                </div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-circle">3</div>
                <div>
                    <div class="step-text-title">View & Explore</div>
                    <div class="step-text-desc">Preview your 3D creation</div>
                </div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-circle">4</div>
                <div>
                    <div class="step-text-title">Download</div>
                    <div class="step-text-desc">Get your 3D asset</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    gen_c1, gen_c2 = st.columns([2.5, 1.5])
    with gen_c2:
        if st.button("🚀 Generate 3D Model →", type="primary", use_container_width=True, key="btn_generate_main_bottom"):
            execute_reconstruction()


# ==================================================
# VIEW: SKETCH STUDIO
# ==================================================
elif st.session_state.nav_page == "Sketch Studio":

    st.markdown("""
    <div class="saas-card" style="padding: 18px 24px;">
        <div class="card-header-row" style="margin: 0;">
            <div>
                <h3 class="card-title-text" style="font-size: 18px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 20h9"></path>
                        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                    </svg>
                    <span>Sketch Studio Workspace</span>
                </h3>
                <p style="font-size: 13px; color: #64748B; margin: 4px 0 0 0;">
                    Interactive digital drawing canvas, brush controls, and real-time computer vision preprocessing.
                </p>
            </div>
            <div class="status-pill">● Active Studio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st_col_left, st_col_right = st.columns([1.1, 0.9], gap="medium")

    with st_col_left:
        st.markdown("""
        <div class="saas-card">
            <div class="card-header-row">
                <div class="card-title-text">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <span>Digital Drawing Canvas</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        tool_col1, tool_col2, tool_col3 = st.columns([1.1, 1.2, 0.9])
        with tool_col1:
            studio_tool = st.radio("Tool", ["✏️ Pen", "🧹 Eraser"], horizontal=True, key="studio_tool_select", label_visibility="collapsed")
        with tool_col2:
            st.caption("Stroke Width")
            studio_width = st.slider("Stroke Width", 1, 24, 4, key="studio_width_slider", label_visibility="collapsed")
        with tool_col3:
            st.caption("Brush Color")
            studio_color = st.color_picker("Brush Color", "#000000", key="studio_color_picker", label_visibility="collapsed") if "Pen" in studio_tool else "#FFFFFF"

        from streamlit_drawable_canvas import st_canvas
        studio_canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=studio_width,
            stroke_color=studio_color,
            background_color="#FFFFFF",
            height=360,
            width=540,
            drawing_mode="freedraw",
            key=f"studio_canvas_{st.session_state.canvas_key}",
            return_image_data=True
        )

        if studio_canvas.image_data is not None:
            arr = studio_canvas.image_data.astype(np.uint8)
            if np.any(arr[:, :, :3] < 250):
                drawn_img = Image.fromarray(arr, mode="RGBA").convert("RGB")
                st.session_state.selected_sketch = drawn_img
                st.session_state.sketch_source_name = "Studio Freehand Concept"

        c_act1, c_act2, c_act3 = st.columns(3)
        with c_act1:
            if st.button("🗑️ Clear Canvas", use_container_width=True, key="studio_clear_btn"):
                st.session_state.canvas_key += 1
                st.session_state.selected_sketch = None
                st.session_state.sketch_source_name = None
                st.session_state.processed_results = None
                st.rerun()

        with c_act2:
            if st.session_state.selected_sketch is not None:
                buf = io.BytesIO()
                st.session_state.selected_sketch.save(buf, format="PNG")
                st.download_button("💾 Download Sketch", data=buf.getvalue(), file_name="studio_sketch.png", mime="image/png", use_container_width=True)

        with c_act3:
            if st.button("🔬 Preprocess Sketch", use_container_width=True, key="studio_run_preprocess_btn"):
                if st.session_state.selected_sketch is not None:
                    temp_p = os.path.join(OUTPUT_DIR, "studio_input.png")
                    st.session_state.selected_sketch.save(temp_p)
                    res = preprocess_sketch(temp_p, PREPROCESS_DIR)
                    st.session_state.processed_results = res
                    st.session_state.sketch_metrics = get_sketch_metrics(temp_p)
                    st.rerun()
                else:
                    st.warning("Draw a concept on the canvas first!")

        st.markdown("</div>", unsafe_allow_html=True)

    with st_col_right:
        active_model = st.session_state.current_model_path

        render_3d_viewer(
            model_file_path=active_model,
            height=495,
            model_title="3D Model Preview"
        )

        if active_model and os.path.exists(active_model):
            model_base = os.path.basename(active_model).replace(".glb", "").replace(".obj", "").replace("_", " ").title()
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                glb_file = active_model if active_model.endswith(".glb") else active_model.replace(".obj", ".glb")
                if os.path.exists(glb_file):
                    with open(glb_file, "rb") as fg:
                        st.download_button("📥 Download GLB", data=fg, file_name=f"{model_base.lower().replace(' ', '_')}.glb", mime="model/gltf-binary", use_container_width=True)
            with d_c2:
                obj_file = active_model if active_model.endswith(".obj") else active_model.replace(".glb", ".obj")
                if os.path.exists(obj_file):
                    with open(obj_file, "rb") as fo:
                        st.download_button("📥 Download Wavefront OBJ", data=fo, file_name=f"{model_base.lower().replace(' ', '_')}.obj", mime="text/plain", use_container_width=True)

    # PREPROCESSING INSPECTION SECTION
    if st.session_state.processed_results is not None:
        st.markdown("""
        <div class="saas-card">
            <div class="card-header-row">
                <div class="card-title-text">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 12h20"></path>
                        <path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6"></path>
                    </svg>
                    <span>Computer Vision Pipeline Inspection (OpenCV)</span>
                </div>
                <div class="status-pill">Denoised & Vectorized</div>
            </div>
        """, unsafe_allow_html=True)

        cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
        with cv_col1:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#0F172A; margin-bottom:6px;'>1. Original Concept</div>", unsafe_allow_html=True)
            if st.session_state.selected_sketch:
                st.image(st.session_state.selected_sketch, use_container_width=True)

        with cv_col2:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#0F172A; margin-bottom:6px;'>2. Gaussian Denoised</div>", unsafe_allow_html=True)
            if os.path.exists(st.session_state.processed_results["denoised"]):
                st.image(st.session_state.processed_results["denoised"], use_container_width=True)

        with cv_col3:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#0F172A; margin-bottom:6px;'>3. Otsu Binary Mask</div>", unsafe_allow_html=True)
            if os.path.exists(st.session_state.processed_results["binary"]):
                st.image(st.session_state.processed_results["binary"], use_container_width=True)

        with cv_col4:
            st.markdown("<div style='font-size:12px; font-weight:700; color:#0F172A; margin-bottom:6px;'>4. Technical Metrics</div>", unsafe_allow_html=True)
            m = st.session_state.sketch_metrics or {}
            st.markdown(f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; font-size:12px;">
                <div style="margin-bottom:6px;"><strong>Dimensions:</strong> {m.get('width', 0)} × {m.get('height', 0)} px</div>
                <div style="margin-bottom:6px;"><strong>Aspect Ratio:</strong> {m.get('aspect_ratio', 1.0)}</div>
                <div style="margin-bottom:6px;"><strong>Stroke Density:</strong> {m.get('stroke_density', 0)}%</div>
                <div style="margin-bottom:6px;"><strong>Edge Points:</strong> {m.get('edge_points', 0)}</div>
                <div><strong>Contours:</strong> {m.get('contour_count', 0)} closed</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Studio Bottom Action
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    st_act1, st_act2 = st.columns([2.5, 1.5])
    with st_act2:
        if st.button("🚀 Reconstruct 3D Model Now →", type="primary", use_container_width=True, key="studio_reconstruct_btn"):
            execute_reconstruction()


# ==================================================
# VIEW: MY CREATIONS
# ==================================================
elif st.session_state.nav_page == "My Creations":

    st.markdown("""
    <div class="saas-card">
        <div class="card-header-row">
            <h3 class="card-title-text">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                </svg>
                <span>My Creations & Model Gallery</span>
            </h3>
            <div class="status-pill">Local Session Archive</div>
        </div>
        <p style="font-size:13px; color:#64748B; margin:0;">
            Review, re-inspect in 3D, and download previously generated 3D assets (GLB and OBJ).
        </p>
    </div>
    """, unsafe_allow_html=True)

    creations = get_all_creations()

    if not creations:
        st.markdown("""
        <div class="saas-card" style="text-align:center; padding:44px 20px;">
            <div style="margin-bottom:14px;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
            </div>
            <h4 style="font-size:16px; font-weight:700; color:#0F172A; margin:0 0 6px 0;">No Reconstructed 3D Assets Yet</h4>
            <p style="font-size:13px; color:#64748B; max-width:440px; margin:0 auto 18px auto;">
                Create or upload a sketch on the Home page or Sketch Studio, then click 'Generate 3D Model' to produce your first 3D asset.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c_go1, c_go2, c_go3 = st.columns([1.5, 1, 1.5])
        with c_go2:
            if st.button("🚀 Go to Studio & Create", type="primary", use_container_width=True):
                st.session_state.nav_page = "Home"
                st.rerun()
    else:
        grid_cols = st.columns(3)
        for i, c in enumerate(creations):
            col_target = grid_cols[i % 3]
            with col_target:
                st.markdown("""
                <div class="saas-card" style="padding:16px;">
                """, unsafe_allow_html=True)

                if c.get("thumb_path") and os.path.exists(c["thumb_path"]):
                    st.image(c["thumb_path"], use_container_width=True)
                else:
                    st.markdown("<div style='height:120px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#94A3B8; font-size:12px;'>No Thumbnail</div>", unsafe_allow_html=True)

                engine_tag = c.get("engine", "OpenCV Fallback")
                badge_bg = "#EFF6FF" if "TripoSR" in engine_tag else "#F1F5F9"
                badge_fg = "#2563EB" if "TripoSR" in engine_tag else "#475569"

                st.markdown(f"""
                <h4 style="font-size:15px; font-weight:700; margin:10px 0 4px 0; color:#0F172A;">{c.get('title', 'Untitled')}</h4>
                <div style="font-size:11px; color:#64748B; margin-bottom:8px;">📅 {c.get('date', 'Recent')}</div>
                <div style="display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap;">
                    <span style="background:{badge_bg}; color:{badge_fg}; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;">{engine_tag}</span>
                    <span style="background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:11px;">{c.get('object_type', 'Object')}</span>
                </div>
                """, unsafe_allow_html=True)

                active_file = c.get("glb_path") or c.get("obj_path")

                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if active_file and os.path.exists(active_file):
                        if st.button("👁️ Inspect 3D", key=f"inspect_{c['id']}", use_container_width=True):
                            st.session_state.current_model_path = active_file
                            st.session_state.current_model_format = "glb" if active_file.endswith(".glb") else "obj"
                            st.session_state.current_engine = c.get("engine", "Archived Model")
                            if c.get("thumb_path") and os.path.exists(c["thumb_path"]):
                                st.session_state.selected_sketch = Image.open(c["thumb_path"])
                            st.session_state.object_prompt = c.get("prompt", "")
                            st.session_state.nav_page = "Home"
                            st.rerun()

                with btn_c2:
                    if c.get("glb_path") and os.path.exists(c["glb_path"]):
                        with open(c["glb_path"], "rb") as cf_glb:
                            st.download_button("📥 GLB", data=cf_glb, file_name=f"{c['id']}.glb", mime="model/gltf-binary", key=f"dl_glb_{c['id']}", use_container_width=True)
                    elif c.get("obj_path") and os.path.exists(c["obj_path"]):
                        with open(c["obj_path"], "rb") as cf_obj:
                            st.download_button("📥 OBJ", data=cf_obj, file_name=f"{c['id']}.obj", mime="text/plain", key=f"dl_obj_{c['id']}", use_container_width=True)

                if st.button("🗑️ Delete", key=f"del_{c['id']}", use_container_width=True):
                    delete_creation(c["id"])
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# VIEW: HOW IT WORKS
# ==================================================
elif st.session_state.nav_page == "How It Works":

    st.markdown("""
    <div class="saas-card">
        <div class="card-header-row">
            <h3 class="card-title-text">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polygon points="10 8 16 12 10 16 10 8"></polygon>
                </svg>
                <span>System Architecture & Reconstruction Pipeline</span>
            </h3>
            <div class="status-pill">TripoSR + OpenCV Pipeline</div>
        </div>
        <p style="font-size:13px; color:#475569; line-height:1.6; margin:0;">
            SKETCH2SHAPE AI integrates deep learning neural field reconstruction with deterministic computer vision failover.
        </p>
    </div>
    """, unsafe_allow_html=True)

    hw_cols = st.columns(4)
    steps = [
        ("01", "Sketch Input", "Upload or Draw", "High-resolution PNG/JPG scans or live manual drawing via HTML5 digital canvas with stroke controls.", "✏️"),
        ("02", "Image Preprocessing", "OpenCV Engine", "Gaussian spatial filtering, Otsu adaptive thresholding, and contour vectorization.", "🔬"),
        ("03", "TripoSR Neural Field", "Tesla T4 GPU", "Triplane transformer predicting 3D signed distance fields and Marching Cubes in ~1.5s.", "🧠"),
        ("04", "Asset Export & WebGL", "Interactive 3D", "Three.js WebGL viewport rendering GLB (vertex colors & PBR) and Wavefront OBJ.", "📦")
    ]

    for idx, (num, title, sub, desc, icon) in enumerate(steps):
        with hw_cols[idx]:
            st.markdown(f"""
            <div class="saas-card" style="height:100%; min-height:240px;">
                <div style="font-size:11px; font-weight:800; color:#2563EB; margin-bottom:4px;">STAGE {num}</div>
                <div style="font-size:28px; margin-bottom:8px;">{icon}</div>
                <h4 style="font-size:15px; font-weight:700; color:#0F172A; margin:0 0 2px 0;">{title}</h4>
                <div style="font-size:11px; font-weight:600; color:#2563EB; margin-bottom:8px;">{sub}</div>
                <p style="font-size:12px; color:#64748B; line-height:1.5; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Engine Comparison Table
    st.markdown("""
    <div class="saas-card" style="margin-top:10px;">
        <h4 style="font-size:15px; font-weight:700; color:#0F172A; margin:0 0 14px 0;">
            ⚖️ Pipeline Engine Comparison: Neural Field vs. Geometric Fallback
        </h4>
        <div style="overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:12px; text-align:left;">
                <thead>
                    <tr style="background:#F8FAFC; border-bottom:2px solid #E2E8F0;">
                        <th style="padding:10px 14px; font-weight:700; color:#0F172A;">Specification / Capability</th>
                        <th style="padding:10px 14px; font-weight:700; color:#2563EB;">TripoSR Neural Engine (Cloud GPU)</th>
                        <th style="padding:10px 14px; font-weight:700; color:#475569;">OpenCV Extrusion Engine (Local Fallback)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">Underlying Technology</td>
                        <td style="padding:10px 14px; color:#334155;">Triplane Transformer + Marching Cubes SDF</td>
                        <td style="padding:10px 14px; color:#64748B;">Otsu Adaptive Threshold + Contour Polygonization</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0; background:#FAFCFF;">
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">3D Surface Geometry</td>
                        <td style="padding:10px 14px; color:#334155;">Full continuous 3D watertight surface mesh</td>
                        <td style="padding:10px 14px; color:#64748B;">Extruded 2.5D closed polygon boundary</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">Color & Shading</td>
                        <td style="padding:10px 14px; color:#334155;">Neural baked vertex colors & PBR materials</td>
                        <td style="padding:10px 14px; color:#64748B;">Smooth studio Gouraud / Lambert shading</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0; background:#FAFCFF;">
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">Hardware Requirement</td>
                        <td style="padding:10px 14px; color:#334155;">NVIDIA Tesla T4 GPU (~14.56 GB VRAM)</td>
                        <td style="padding:10px 14px; color:#64748B;">Zero-GPU required (runs on any Intel CPU)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">Inference Speed</td>
                        <td style="padding:10px 14px; color:#334155;">~1.5 – 2.5 seconds</td>
                        <td style="padding:10px 14px; color:#64748B;">&lt; 0.15 seconds (instantaneous)</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 14px; font-weight:600; color:#0F172A;">Export Compatibility</td>
                        <td style="padding:10px 14px; color:#334155;">Binary glTF (.glb) & Wavefront OBJ (.obj)</td>
                        <td style="padding:10px 14px; color:#64748B;">Standard Wavefront OBJ (.obj)</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# VIEW: ABOUT PROJECT
# ==================================================
elif st.session_state.nav_page == "About Project":

    st.markdown("""
    <div class="saas-card">
        <div class="card-header-row">
            <h3 class="card-title-text">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span>Project Information & Hackathon Details</span>
            </h3>
            <div class="status-pill">TechZephyr 4.0</div>
        </div>

        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:14px; margin-top:12px;">
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:10px;">
                <div style="font-size:11px; text-transform:uppercase; color:#64748B; font-weight:700;">Project Name</div>
                <div style="font-size:15px; font-weight:700; color:#0F172A; margin-top:3px;">SKETCH2SHAPE AI</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:10px;">
                <div style="font-size:11px; text-transform:uppercase; color:#64748B; font-weight:700;">Hackathon Event</div>
                <div style="font-size:15px; font-weight:700; color:#2563EB; margin-top:3px;">TechZephyr 4.0</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:10px;">
                <div style="font-size:11px; text-transform:uppercase; color:#64748B; font-weight:700;">Host Institution</div>
                <div style="font-size:15px; font-weight:700; color:#0F172A; margin-top:3px;">IIT Bhubaneswar</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:10px;">
                <div style="font-size:11px; text-transform:uppercase; color:#64748B; font-weight:700;">Domain & Track</div>
                <div style="font-size:15px; font-weight:700; color:#2563EB; margin-top:3px;">Computer Vision & Generative AI</div>
            </div>
        </div>
    </div>

    <div class="saas-card">
        <h4 style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">Technical Stack & Specifications</h4>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px;">
            <div style="padding:14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">
                <div style="font-weight:700; font-size:13px; color:#1E293B;">Neural 3D Engine</div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">TripoSR (Stability AI & VAST AI Research)</div>
            </div>
            <div style="padding:14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">
                <div style="font-weight:700; font-size:13px; color:#1E293B;">GPU Acceleration</div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">Google Colab NVIDIA Tesla T4 (~14.56 GB VRAM)</div>
            </div>
            <div style="padding:14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">
                <div style="font-weight:700; font-size:13px; color:#1E293B;">On-Device Computer Vision</div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">OpenCV, NumPy, Pillow, Contour Polygonization</div>
            </div>
            <div style="padding:14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">
                <div style="font-weight:700; font-size:13px; color:#1E293B;">Interactive 3D Engine</div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">Three.js WebGL, OrbitControls, GLTFLoader, OBJ Parser</div>
            </div>
        </div>
    </div>

    <div class="saas-card">
        <h4 style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:10px;">Key Architectural Highlights</h4>
        <ul style="font-size:13px; color:#475569; line-height:1.7; margin:0; padding-left:20px;">
            <li><strong>Zero Client GPU Requirement:</strong> Allows standard student laptops (Intel UHD / HD Graphics) to generate and view complex 3D assets effortlessly.</li>
            <li><strong>Failover Resilience:</strong> Automatic failover between Cloud GPU neural field generation and deterministic on-device OpenCV contour extrusion.</li>
            <li><strong>Standard Asset Export:</strong> Produces production-ready <code>.glb</code> and <code>.obj</code> files ready for Blender, Unity, Unreal Engine, and Web 3D environments.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
