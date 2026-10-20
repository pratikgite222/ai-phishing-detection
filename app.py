"""
AI-Based Phishing Website Detection
Streamlit Web Application

A modern, clean, production-style web interface for live URL phishing analysis.
Powered by a Tuned Random Forest Classifier analyzing 54 structural and lexical URL features.

Key Principles:
- Strictly decoupled: Delegates extraction and inference to src/predict.py
- Non-crashing: Comprehensive error handling for empty, malformed, or unreachable URLs
- Educational: Designed for live demonstration and college viva presentation
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predict import load_feature_names, load_model, predict_url

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Phishing Website Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished cybersecurity / AI theme
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 20px;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-med {
        background-color: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FDE047;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .badge-very-high {
        background-color: #7F1D1D;
        color: #FEF2F2;
        border: 1px solid #991B1B;
    }
    .disclaimer-box {
        font-size: 0.82rem;
        color: #64748B;
        background-color: #F1F5F9;
        border-left: 4px solid #94A3B8;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MODEL & RESOURCE CACHING
# ==============================================================================
@st.cache_resource(show_spinner="Loading machine learning model...")
def get_cached_model():
    """Cache the trained Random Forest model in Streamlit memory."""
    return load_model("models/random_forest_tuned.joblib")


@st.cache_resource
def get_cached_feature_names():
    """Cache expected feature names list."""
    return load_feature_names("models/feature_names.json")


# Pre-warm model cache
get_cached_model()
expected_feature_names = get_cached_feature_names()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_risk_assessment(phish_prob: float) -> Tuple[str, str, str]:
    """
    Determine risk tier based on phishing probability:
    - 0% - 30%: Low Risk
    - 30% - 60%: Medium Risk
    - 60% - 80%: High Risk
    - 80% - 100%: Very High Risk
    """
    percent = phish_prob * 100
    if percent < 30.0:
        return "Low Risk", "badge-low", "#16A34A"
    elif percent < 60.0:
        return "Medium Risk", "badge-med", "#CA8A04"
    elif percent < 80.0:
        return "High Risk", "badge-high", "#DC2626"
    else:
        return "Very High Risk", "badge-very-high", "#991B1B"


# ==============================================================================
# SIDEBAR: ABOUT THE PROJECT
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛡️ About the Project")
    st.markdown(
        """
        This application uses machine learning to classify websites as **legitimate** or **potentially phishing** based on lexical and structural URL properties.
        
        The model inspects **54 numerical features** extracted from the URL and evaluates them using a **Tuned Random Forest classifier**.
        """
    )

    st.markdown("---")
    st.markdown("#### 📊 Model Evaluation Benchmarks")
    st.markdown(
        """
        *Metrics computed strictly on held-out test partition ($N = 2,286$):*
        - **Test Accuracy**: `91.29%`
        - **Phishing F1-Score**: `91.37%`
        - **Phishing Recall**: `92.13%` *(Missed only 90 of 1,143 phishing test URLs)*
        """
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Technical Architecture")
    st.markdown(
        """
        - **Model**: Tuned Random Forest (200 Trees, `max_features='log2'`)
        - **Number of URL Features**: `54`
        - **Feature Scaling**: Unscaled input
        - **Dataset Size**: `11,430` balanced URLs
        - **Frameworks**: `scikit-learn`, `pandas`, `Streamlit`
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8;">
        Academic Reference: Hannousse & Yahiouche (2021), <i>Engineering Applications of Artificial Intelligence</i>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN PAGE HEADER & INPUT
# ==============================================================================
st.markdown('<div class="main-title">🛡️ AI-Based Phishing Website Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Machine Learning Powered URL Security Analysis</div>', unsafe_allow_html=True)

# Demo Presets for Easy Viva Demonstration
st.markdown("**Sample URLs for Quick Testing:**")
cols_preset = st.columns(5)
preset_urls = [
    ("🌐 Google", "https://www.google.com"),
    ("🛒 Amazon", "https://www.amazon.com"),
    ("⚠️ Suspicious IP", "http://192.168.1.1/login.php"),
    ("🚨 Spoofed PayPal", "http://paypal-login-security.example.com/verify/account"),
    ("🔑 Query Login", "https://example.com/login?user=test&id=123"),
]

# State variable for input field
if "url_input" not in st.session_state:
    st.session_state.url_input = ""

for i, (btn_label, preset_val) in enumerate(preset_urls):
    if cols_preset[i].button(btn_label, key=f"btn_preset_{i}"):
        st.session_state.url_input = preset_val

url_text = st.text_input(
    label="Enter a website URL",
    value=st.session_state.url_input,
    placeholder="https://example.com",
    help="Enter a complete or partial URL (e.g., https://example.com/path?query=val)",
)

col_btn, col_network = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🔍 Analyze Website", type="primary", use_container_width=True)

with col_network:
    check_network = st.checkbox(
        "HTTP redirect check with a 1.5-second timeout",
        value=True,
        help="Checks live HTTP redirection hops (nb_redirection). Safely defaults to 0 if offline or unreachable.",
    )

st.markdown("---")


# ==============================================================================
# PREDICTION EXECUTION & DISPLAY
# ==============================================================================
if analyze_btn:
    if not url_text or not url_text.strip():
        st.warning("⚠️ Please enter a valid website URL before analyzing.")
    else:
        with st.spinner("Extracting 54 URL features and running Machine Learning inference..."):
            try:
                # Call prediction pipeline (NO DUPLICATE LOGIC)
                result = predict_url(
                    url=url_text,
                    check_network=check_network,
                    network_timeout=1.5,
                    include_features=True,
                )

                pred = result["prediction"]
                label = result["label"]
                phish_prob = result["phishing_probability"]
                legit_prob = result["legitimate_probability"]
                norm_url = result["normalized_url"]
                features = result.get("features", {})

                risk_title, badge_class, progress_color = get_risk_assessment(phish_prob)

                # 1. Main Prediction Banner
                st.markdown("### Prediction Result")
                if pred == 0:
                    st.success("### ✅ Legitimate Website")
                else:
                    st.error("### ⚠️ Phishing Website")

                # 2. Risk Metrics Display
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Phishing Risk</div>
                            <div class="metric-value" style="color: {progress_color};">{phish_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Legitimate Confidence</div>
                            <div class="metric-value" style="color: #16A34A;">{legit_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Risk Category</div>
                            <div style="margin-top: 10px;">
                                <span class="badge-pill {badge_class}">{risk_title}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Progress Bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.write("**Model Phishing Probability Distribution:**")
                st.progress(min(1.0, max(0.0, phish_prob)))

                # Disclaimer Box
                st.markdown(
                    """
                    <div class="disclaimer-box">
                    ℹ️ <b>Model Risk Assessment Notice:</b> This result is an automated probability estimate generated by a machine learning model analyzing 54 structural URL properties. It does not provide an absolute or legal guarantee of website safety.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # 3. URL Information
                st.markdown("### 🌐 URL Information")
                col_u1, col_u2, col_u3 = st.columns([3, 3, 2])
                with col_u1:
                    st.write(f"**Entered URL:** `{url_text}`")
                with col_u2:
                    st.write(f"**Normalized URL:** `{norm_url}`")
                with col_u3:
                    st.write(f"**Features Analyzed:** `54`")

                st.markdown("---")

                # 4. Technical Analysis (Important Key Features)
                with st.expander("🔎 Technical Analysis (Key Extracted Features)", expanded=True):
                    st.markdown("Selected core indicators extracted from the URL:")
                    
                    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                    with tcol1:
                        st.metric("URL Length", int(features.get("length_url", 0)))
                        st.metric("Hostname Length", int(features.get("length_hostname", 0)))
                        st.metric("Dots Count", int(features.get("nb_dots", 0)))
                        st.metric("Hyphens Count", int(features.get("nb_hyphens", 0)))
                        st.metric("@ Symbol Count", int(features.get("nb_at", 0)))

                    with tcol2:
                        st.metric("IP Address Detected", "Yes (1)" if features.get("ip") == 1 else "No (0)")
                        st.metric("Query Parameters", int(features.get("nb_qm", 0)))
                        st.metric("Subdomain Levels", int(features.get("nb_subdomains", 0)))
                        st.metric("Abnormal Subdomain", "Yes (1)" if features.get("abnormal_subdomain") == 1 else "No (0)")
                        st.metric("Random Domain", "Yes (1)" if features.get("random_domain") == 1 else "No (0)")

                    with tcol3:
                        st.metric("Shortening Service", "Yes (1)" if features.get("shortening_service") == 1 else "No (0)")
                        st.metric("Executable / Script Path", "Yes (1)" if features.get("path_extension") == 1 else "No (0)")
                        st.metric("HTTP Redirection Hops", int(features.get("nb_redirection", 0)))
                        st.metric("External Redirections", int(features.get("nb_external_redirection", 0)))
                        st.metric("Suspicious TLD", "Yes (1)" if features.get("suspecious_tld") == 1 else "No (0)")

                    with tcol4:
                        st.metric("Phish Hints Count", int(features.get("phish_hints", 0)))
                        st.metric("Domain in Brand", "Yes (1)" if features.get("domain_in_brand") == 1 else "No (0)")
                        st.metric("Brand in Subdomain", "Yes (1)" if features.get("brand_in_subdomain") == 1 else "No (0)")
                        st.metric("Brand in Path", "Yes (1)" if features.get("brand_in_path") == 1 else "No (0)")
                        st.metric("Double Slash in Path", int(features.get("nb_dslash", 0)))

                # 5. Complete 54-Feature Vector View
                with st.expander("📊 View All 54 Extracted Features (Complete Feature Vector)"):
                    st.markdown("Full 54-dimensional feature vector in exact model alignment:")
                    feature_rows = []
                    for idx, fname in enumerate(expected_feature_names, 1):
                        feature_rows.append({
                            "#": idx,
                            "Feature Name": fname,
                            "Extracted Value": features.get(fname, "N/A"),
                        })
                    df_features = pd.DataFrame(feature_rows)
                    st.dataframe(df_features, use_container_width=True, height=450)

            except ValueError as ve:
                st.error(f"⚠️ Validation Error: {ve}")
            except Exception as e:
                st.error(f"❌ Unexpected Error during analysis: {e}")
                st.info("The application caught this exception safely without crashing.")
