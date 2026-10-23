"""
================================================================================
Breast Cancer AI Analysis - UI Design System & Reusable Components (V8 Final)
================================================================================
Modular visual styling and UI components for Streamlit medical research dashboard:
- Dark glassmorphism & medical theme styling
- Streamlit toolbar & chrome removal
- Hero header & system telemetry banner
- Prediction cards (Benign / Malignant with research prototype wording)
- End-to-end model pipeline flow diagram (HTML / SVG)
- Mandatory medical research safety banners
================================================================================
"""

import streamlit as st
from typing import Dict, Any, Optional
from app.config import SAFETY_DISCLAIMER_SHORT, SAFETY_DISCLAIMER_FULL


def inject_custom_css():
    """Injects high-end medical AI CSS styling and cleans up Streamlit UI artifacts."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Remove Streamlit default toolbar, header, deploy button, status widgets */
        header,
        header[data-testid="stHeader"],
        .stAppHeader,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stAppToolbar,
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidgetIcon"],
        [data-testid="stMainMenu"],
        #MainMenu,
        footer,
        .stDeployButton,
        button[aria-label="Stop"] {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
            width: 0px !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        
        /* Main Container Spacing */
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1400px;
        }
        
        /* Hero Header Card */
        .hero-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 24px 32px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.15);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin: 0;
            background: linear-gradient(to right, #ffffff, #93c5fd, #5eead4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            font-weight: 400;
            color: #cbd5e1;
            margin-top: 6px;
            margin-bottom: 14px;
        }
        .hero-badge-container {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .hero-badge {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #f1f5f9;
            letter-spacing: 0.02em;
        }
        
        /* Prediction Result Cards */
        .result-card-malignant {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(153, 27, 27, 0.22) 100%);
            border: 2px solid #ef4444;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.18);
            margin-bottom: 20px;
        }
        .result-card-benign {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 95, 70, 0.22) 100%);
            border: 2px solid #10b981;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.18);
            margin-bottom: 20px;
        }
        .result-title-malignant {
            color: #ef4444;
            font-size: 2.0rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: 0.02em;
        }
        .result-title-benign {
            color: #10b981;
            font-size: 2.0rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: 0.02em;
        }
        .result-subtext {
            font-size: 0.95rem;
            color: #64748b;
            margin-top: 6px;
            font-weight: 500;
        }
        
        /* Research Disclaimer Banner */
        .disclaimer-box {
            background: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 20px;
        }
        
        /* Metric & Spec Badges */
        .stat-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .stat-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .stat-val {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Pipeline Flowchart Container */
        .pipeline-flow {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            color: #f8fafc;
        }
    </style>
    """, unsafe_allow_html=True)


def render_hero_header():
    """Renders top header across all pages."""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">AI-Assisted Breast Cancer Research Prototype</div>
        <div class="hero-subtitle">
            Wiener Filtering + Dual-Resolution U-KAN + Multi-Scale ViT (V8 Frozen Final Model)
        </div>
        <div class="hero-badge-container">
            <span class="hero-badge">🧬 U-KAN Spline Segmentation (512+768)</span>
            <span class="hero-badge">📐 Dual-Res Top-3 RoI Proposals</span>
            <span class="hero-badge">🌐 Full Mammogram Global Context</span>
            <span class="hero-badge">✨ High-Frequency Calcification Filter</span>
            <span class="hero-badge">⚡ V8 ViT Arm F Joint Fusion (4608-d)</span>
            <span class="hero-badge" style="background: rgba(16, 185, 129, 0.25); border-color: #10b981;">🔒 Final Model: V8 Frozen</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_safety_disclaimer(mode: str = "banner"):
    """Renders mandatory medical research safety notices."""
    if mode == "banner":
        st.markdown(f"""
        <div class="disclaimer-box">
            <b>Research Prototype Notice:</b><br>
            {SAFETY_DISCLAIMER_SHORT}
        </div>
        """, unsafe_allow_html=True)
    elif mode == "full":
        st.markdown(f"""
        <div class="disclaimer-box" style="font-size: 0.95rem; padding: 20px; line-height: 1.6;">
            <b>IMPORTANT MEDICAL RESEARCH NOTICE:</b><br>
            {SAFETY_DISCLAIMER_FULL}
        </div>
        """, unsafe_allow_html=True)


def render_prediction_card(predicted_class: str, p_malignant: float, p_benign: float):
    """
    Renders prediction card with cautious language and probabilities.
    Never uses definitive diagnostic claims.
    """
    if predicted_class.upper() == "MALIGNANT":
        st.markdown(f"""
        <div class="result-card-malignant">
            <div class="result-title-malignant">AI Research Prediction: MALIGNANT</div>
            <div class="result-subtext">AI-Assisted Research Classification • Elevated Risk Morphology Detected</div>
            <div style="margin-top: 14px; font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                Malignant Probability: <span style="color: #ef4444; font-family: 'JetBrains Mono', monospace;">{p_malignant * 100:.2f}%</span> &nbsp;|&nbsp; 
                Benign Probability: <span style="color: #10b981; font-family: 'JetBrains Mono', monospace;">{p_benign * 100:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-benign">
            <div class="result-title-benign">AI Research Prediction: BENIGN</div>
            <div class="result-subtext">AI-Assisted Research Classification • Low Suspicion Morphology Detected</div>
            <div style="margin-top: 14px; font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                Benign Probability: <span style="color: #10b981; font-family: 'JetBrains Mono', monospace;">{p_benign * 100:.2f}%</span> &nbsp;|&nbsp; 
                Malignant Probability: <span style="color: #ef4444; font-family: 'JetBrains Mono', monospace;">{p_malignant * 100:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_pipeline_flowchart():
    """Renders the step-by-step visual model pipeline diagram."""
    st.markdown("""
    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 14px; padding: 24px; color: #f8fafc; font-family: 'Inter', sans-serif;">
        <h4 style="color: #38bdf8; margin-top: 0; margin-bottom: 16px;">V8 Final Architecture: End-to-End Pipeline Flow</h4>
        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; font-size: 0.95rem; font-weight: 500;">
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                📷 <b>Input Mammogram</b> (Screening / Diagnostic FFDM: PNG, JPG, BMP, DICOM)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                🌊 <b>Adaptive Wiener Filtering</b> (5×5 Local Spatial Denoising)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                ✨ <b>CLAHE & Normalization</b> (Contrast Limited Adaptive Histogram Equalization → [0.0, 1.0])
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #0284c7; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center; box-shadow: 0 0 12px rgba(2, 132, 199, 0.3);">
                🧬 <b>Dual-Resolution U-KAN Segmentation</b> (512×512 + 768×768 Kolmogorov-Arnold Spline Bottleneck)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                🎯 <b>V8 Candidate Proposal Generation</b> (Deterministic Scoring: Area, Peak Prob, Compactness, Contrast; K=3)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                📐 <b>Multi-Scale Concentric RoI Extraction</b> (Local 1.2×, Medium 1.8×, Large 2.5×, Context 4.0×)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                🔬 <b>Calcification Patch + Full Mammogram Context</b> (High-Frequency Filter + 224×224 Global Canvas)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #059669; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center; box-shadow: 0 0 12px rgba(5, 150, 105, 0.3);">
                🧠 <b>V8 Vision Transformer (Arm F)</b> (Candidate Softmax Attention + Last 4 ViT Blocks Tuned → 4608-d MLP)
            </div>
            <div style="color: #38bdf8; font-size: 1.2rem;">↓</div>
            <div style="background: #1e293b; border: 1px solid #475569; padding: 10px 24px; border-radius: 8px; width: 80%; text-align: center;">
                📊 <b>AI-Assisted Research Prediction & Attention Rollout</b> (Benign / Malignant Research Output)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
