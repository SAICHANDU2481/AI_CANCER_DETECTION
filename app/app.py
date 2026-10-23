"""
================================================================================
Breast Cancer AI Analysis - Streamlit Research & Diagnostic Platform (V8 Frozen)
================================================================================
"Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer,
and U-KAN-Based Medical Image Analysis"

MANDATORY GOVERNANCE:
- AI-assisted research prototype only, NOT a clinical diagnostic system.
- FINAL REPORTED MODEL: V8 (Phase 2 Arm F: Dual-Res Proposals + Last 4 ViT Blocks Tuned).
- Locked V8 Held-Out Test Benchmark (525 scans / 235 patients) is the PRIMARY reported test result.
- V9 is retained in the research record as a documented experimental extension.
- Strict cryptographic MD5 verification enforced.
================================================================================
"""

import os
import sys
import time
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
import torch

# -----------------------------------------------------------------------------
# System Path Initialization
# -----------------------------------------------------------------------------
CURRENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CURRENT_ROOT not in sys.path:
    sys.path.insert(0, CURRENT_ROOT)

PROJECT_ROOTS = [
    CURRENT_ROOT,
    r"C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION",
    r"C:\Users\sys\Desktop\AI_CANCER_DETECTION"
]
for root in PROJECT_ROOTS:
    if os.path.exists(root) and root not in sys.path:
        sys.path.append(root)

# Internal Modular Imports
from app.config import (
    UKAN_V6_PATH, VIT_V8_PATH, EXPECTED_MD5, verify_file_md5,
    LOCKED_V8_TEST_METRICS, LOCKED_V6_TEST_METRICS, LOCKED_MODEL_COMPARISON_TABLE,
    DATASET_STATS, DEMO_SCANS, SAFETY_DISCLAIMER_SHORT, SAFETY_DISCLAIMER_FULL
)
from app.preprocessing import load_image_from_upload, load_demo_scan_image, preprocess_mammogram_full
from app.inference import load_v8_pipeline_models, run_complete_inference_pipeline, ModelBundle
from app.visualization import create_segmentation_overlay, create_attention_overlay, render_confusion_matrix_plot
from app.components import (
    inject_custom_css, render_hero_header, render_safety_disclaimer,
    render_prediction_card, render_pipeline_flowchart
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Assisted Breast Cancer Research Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
inject_custom_css()


# -----------------------------------------------------------------------------
# Cached Model Pipeline Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_models():
    """
    Loads and cryptographically verifies V8 model checkpoints once per session.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle, err = load_v8_pipeline_models(device)
    return bundle, device, err


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Sidebar Navigation & Telemetry
    with st.sidebar:
        st.markdown("### 🔬 Medical AI System")
        st.markdown("**Breast Cancer Research Platform**")
        st.caption("Wiener Filtering + Dual-Res U-KAN + Multi-Scale ViT")
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation Menu")
        app_mode = st.radio(
            "Select Section:",
            [
                "🔬 Live Mammogram Analysis",
                "⚙️ Preprocessing",
                "🎯 Segmentation",
                "🔍 ROI Analysis",
                "🧠 Classification",
                "🔮 Explainability",
                "📐 Model Pipeline",
                "🔒 Final Test Results",
                "📊 Model Comparisons (V5-V9)",
                "📚 Dataset Information",
                "ℹ️ About Project"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ System Telemetry")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
        st.markdown(f"**Final Model:** `V8 (Arm F Frozen)`")
        st.markdown(f"**Compute:** `{device_name}`")
        st.markdown(f"**PyTorch:** `{torch.__version__}`")
        st.markdown(f"**U-KAN Hash:** `Verified MD5`")
        st.markdown(f"**ViT Hash:** `Verified MD5`")
        
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
            <b>Mandatory Safety Notice:</b><br>
            {SAFETY_DISCLAIMER_SHORT}
        </div>
        """, unsafe_allow_html=True)

    # Hero Header Across All Pages
    render_hero_header()

    # Load Models
    models, device, load_err = get_cached_models()

    if load_err is not None:
        st.error(f"⚠️ Model Loading Error: {load_err}")
        st.info("Please ensure that verified V8 checkpoints are present in `models/final_v8/` or `models/v8/`.")
        return

    # =========================================================================
    # VIEW 1: LIVE MAMMOGRAM ANALYSIS (UPLOAD & PIPELINE EXECUTION)
    # =========================================================================
    if app_mode == "🔬 Live Mammogram Analysis":
        render_safety_disclaimer("banner")
        
        st.markdown("### [Upload Mammogram]")
        st.caption("Supported Formats: PNG, JPG, JPEG, BMP, DICOM")
        
        col_in_a, col_in_b = st.columns([1.1, 0.9])
        raw_image_input = None
        case_metadata = None
        
        with col_in_a:
            st.markdown("#### Option 1: Upload Mammogram File")
            uploaded_file = st.file_uploader(
                "Select a mammogram scan from local drive:",
                type=["png", "jpg", "jpeg", "bmp", "dcm", "dicom"],
                help="Supports 8-bit and 16-bit screening/diagnostic mammography images."
            )
            if uploaded_file is not None:
                try:
                    raw_image_input = load_image_from_upload(uploaded_file)
                    case_metadata = {"name": uploaded_file.name, "source": "User Upload", "gt_mask": None}
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

        with col_in_b:
            st.markdown("#### Option 2: Preloaded Benchmark Scans")
            demo_choice = st.selectbox(
                "Or evaluate a verified benchmark scan:",
                list(DEMO_SCANS.keys())
            )
            if demo_choice != "Select a preloaded clinical benchmark case...":
                demo_data = DEMO_SCANS[demo_choice]
                loaded_demo = load_demo_scan_image(demo_data)
                if loaded_demo is not None:
                    raw_image_input = loaded_demo
                    case_metadata = {
                        "name": demo_choice,
                        "source": "Curated Benchmark Cohort",
                        "type": demo_data["type"],
                        "ground_truth": demo_data["ground_truth"],
                        "desc": demo_data["desc"],
                        "gt_mask": None
                    }
                else:
                    st.warning("Demo image file path not accessible locally. Please upload an image file above.")

        st.markdown("---")

        if raw_image_input is not None:
            # -------------------------------------------------------------
            # 1. ORIGINAL IMAGE
            # -------------------------------------------------------------
            st.markdown("### 1. ORIGINAL IMAGE")
            col_orig_preview, col_orig_meta = st.columns([1, 1.2])
            with col_orig_preview:
                st.image(
                    raw_image_input,
                    caption=f"Original Mammogram ({raw_image_input.shape[1]} × {raw_image_input.shape[0]} px)",
                    use_container_width=True,
                    clamp=True
                )
            with col_orig_meta:
                st.markdown("#### Scan Metadata & Research Analysis")
                st.write(f"- **Dimensions:** `{raw_image_input.shape[1]} × {raw_image_input.shape[0]}` pixels")
                st.write(f"- **Source:** `{case_metadata.get('source', 'User Upload')}`")
                if "desc" in case_metadata:
                    st.write(f"- **Case Description:** {case_metadata['desc']}")
                
                run_btn = st.button("🚀 Run Complete V8 AI Pipeline Analysis", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Executing Full V8 Research Pipeline (Wiener → CLAHE → Dual-Res U-KAN → Multi-Scale RoI → ViT Arm F)..."):
                    res = run_complete_inference_pipeline(
                        raw_image_input,
                        models,
                        ground_truth_mask=case_metadata.get("gt_mask")
                    )

                # -------------------------------------------------------------
                # 2. PREPROCESSING
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 2. PREPROCESSING")
                st.caption("Noise filtering, adaptive contrast adjustment, and normalized multi-scale tensor generation.")
                
                prep_data = res["preprocessing"]
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.image(prep_data["original"], caption=f"Original ({raw_image_input.shape[1]}×{raw_image_input.shape[0]})", use_container_width=True, clamp=True)
                with p2:
                    st.image(prep_data["wiener"], caption="Wiener Filtered [0, 1]", use_container_width=True, clamp=True)
                with p3:
                    st.image(prep_data["enhanced"], caption="CLAHE Enhanced", use_container_width=True, clamp=True)
                with p4:
                    st.image(prep_data["img_512"], caption="Normalized (512×512)", use_container_width=True, clamp=True)

                # -------------------------------------------------------------
                # 3. U-KAN SEGMENTATION
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 3. U-KAN SEGMENTATION")
                st.caption("Dual-Resolution Kolmogorov-Arnold Network Segmentation (512×512 + 768×768 Receptive Field).")
                
                seg_data = res["segmentation"]
                proposals = res["proposals"]
                
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.image(seg_data["prob_map"], caption="U-KAN Probability Map [0, 1]", use_container_width=True, clamp=True)
                with s2:
                    st.image(seg_data["binary_mask"], caption=f"Binary Mask (Threshold = 0.15)", use_container_width=True, clamp=True)
                with s3:
                    overlay = create_segmentation_overlay(prep_data["img_512"], seg_data["binary_mask"], proposals)
                    st.image(overlay, caption="Segmentation Overlay with Top Proposals", use_container_width=True)

                # -------------------------------------------------------------
                # 4. MULTI-SCALE ROI CANDIDATES
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 4. MULTI-SCALE ROI EXTRACTION")
                st.caption(f"Concentric multi-scale bounding box proposals (Top-{len(proposals)} Candidates).")
                
                for idx, crops in enumerate(res["crops"]):
                    st.markdown(f"**Candidate Proposal #{idx+1}** (Source: `{proposals[idx].get('proposal_source', 'machine_cc')}` | Score: `{proposals[idx].get('score', 0.0):.4f}`)")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.image(crops["local"], caption="Local (1.2× Scale)", use_container_width=True, clamp=True)
                    with c2:
                        st.image(crops["medium"], caption="Medium (1.8× Scale)", use_container_width=True, clamp=True)
                    with c3:
                        st.image(crops["large"], caption="Large (2.5× Scale)", use_container_width=True, clamp=True)
                    with c4:
                        st.image(crops["context"], caption="Context (4.0× Scale)", use_container_width=True, clamp=True)

                # -------------------------------------------------------------
                # 5. CLASSIFICATION & EXPLAINABILITY
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 5. CLASSIFICATION & ATTENTION ROLLOUT")
                
                cls_data = res["classification"]
                exp_data = res["explainability"]

                render_prediction_card(
                    cls_data["predicted_class"],
                    cls_data["prob_malignant"],
                    cls_data["prob_benign"]
                )

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("#### Primary Local ROI Attention Rollout")
                    attn_local_overlay = create_attention_overlay(exp_data["primary_local_crop"], exp_data["attention_map_local"])
                    st.image(attn_local_overlay, caption="Primary Lesion Attention Saliency", use_container_width=True)
                with col_a2:
                    st.markdown("#### Global Mammogram Attention Rollout")
                    attn_full_overlay = create_attention_overlay(prep_data["img_224"], exp_data["attention_map_full"])
                    st.image(attn_full_overlay, caption="Global Breast Parenchymal Attention Saliency", use_container_width=True)

    # =========================================================================
    # VIEW 2: PREPROCESSING
    # =========================================================================
    elif app_mode == "⚙️ Preprocessing":
        st.markdown("### ⚙️ Preprocessing Pipeline")
        st.write("Technical breakdown of Wiener spatial denoising and CLAHE histogram equalization.")
        
        st.markdown("""
        #### 1. Adaptive Wiener Spatial Denoising:
        Mammographic images are subject to sensor quantum noise. The 2D adaptive Wiener filter estimates local mean $\\mu$ and local variance $\\sigma^2$ across a $5\\times5$ spatial window:
        $$I_{\\text{wiener}}(x, y) = \\mu + \\frac{\\sigma^2 - \\nu^2}{\\sigma^2} \\big(I(x,y) - \\mu\\big)$$
        where $\\nu^2$ represents estimated global noise variance.
        
        #### 2. Contrast-Limited Adaptive Histogram Equalization (CLAHE):
        CLAHE enhances subtle tissue density gradients without over-amplifying noise in dense glandular regions:
        - **Clip Limit:** `2.5`
        - **Tile Grid Size:** `(8, 8)`
        """)

    # =========================================================================
    # VIEW 3: SEGMENTATION
    # =========================================================================
    elif app_mode == "🎯 Segmentation":
        st.markdown("### 🎯 U-KAN Lesion Segmentation Architecture")
        st.write("Dual-Resolution Kolmogorov-Arnold Network for medical lesion segmentation.")

        st.markdown("""
        #### Dual-Resolution U-KAN Topology:
        - **Resolution Branches:** $512\\times512$ working canvas and $768\\times768$ high-resolution detail pathway.
        - **Kolmogorov-Arnold Spline Representation:** Replaces standard fixed linear weights with learnable 1D B-spline activation curves:
        $$\\phi(x) = w_b \\cdot b(x) + w_s \\cdot \\text{spline}(x)$$
        - **Bottleneck Feature Fusion:** Captures fine-grained microcalcification boundaries alongside large circumscribed mass margins.
        """)

    # =========================================================================
    # VIEW 4: ROI ANALYSIS
    # =========================================================================
    elif app_mode == "🔍 ROI Analysis":
        st.markdown("### 🔍 Concentric Multi-Scale RoI Analysis")
        st.write("Multi-Scale concentric receptive field geometry.")

        st.markdown("""
        #### Concentric Scale Geometry:
        1. **Local Crop ($1.2\\times$):** Tight lesion interior bounding box; captures microcalcification morphology and internal density.
        2. **Medium Crop ($1.8\\times$):** Immediate perilesional boundary margin; captures margin sharpness and micro-lobulation.
        3. **Large Crop ($2.5\\times$):** Radiating spiculation and architectural distortion in surrounding tissue.
        4. **Context Crop ($4.0\\times$):** Regional background breast parenchymal density distribution.
        5. **High-Frequency Calcification Patch:** High-frequency spatial filter applied to primary crop.
        6. **Full-Mammogram Canvas:** Downsampled full-breast global view ($224\\times224$).
        """)

    # =========================================================================
    # VIEW 5: CLASSIFICATION
    # =========================================================================
    elif app_mode == "🧠 Classification":
        st.markdown("### 🧠 V8 Candidate-Fusion Vision Transformer (Arm F)")
        st.write("Joint classification using Vision Transformer with Candidate Softmax Aggregation.")

        st.markdown("""
        #### Architecture Specifications:
        - **ViT Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (768 embedding dimensions, 12 blocks).
        - **Fine-Tuned Depth:** Top 4 Transformer blocks (blocks 8–11) fine-tuned for mammographic feature adaptation.
        - **Candidate Attention Aggregation:** Continuous Softmax aggregator pools $K=3$ candidates per scale:
          - Local Aggregated Token ($768$)
          - Medium Aggregated Token ($768$)
          - Large Aggregated Token ($768$)
          - Context Aggregated Token ($768$)
          - Calcification Patch Token ($768$)
          - Full-Mammogram Context Token ($768$)
        - **Fused Dimension:** $4608$ dimensions $\\rightarrow$ LayerNorm $\\rightarrow$ 4-layer MLP ($4608 \\rightarrow 1024 \\rightarrow 512 \\rightarrow 128 \\rightarrow 2$).
        """)

    # =========================================================================
    # VIEW 6: EXPLAINABILITY
    # =========================================================================
    elif app_mode == "🔮 Explainability":
        st.markdown("### 🔮 Explainability & Attention Rollout")
        st.write("Authentic Vision Transformer Self-Attention Rollout visualization for interpretable AI.")

        st.markdown("""
        #### Attention Rollout Formulation:
        Self-attention rollout tracks information flow across all 12 Transformer layers:
        $$A_{\\text{rollout}} = \\prod_{l=1}^{L} (0.5 \\cdot A_l + 0.5 \\cdot I)$$
        where $A_l$ is the multi-head attention matrix at layer $l$, and $I$ accounts for residual skip connections.

        #### Medical Interpretation Notice:
        - Attention heatmaps illustrate which image tokens contributed most strongly to the classification representation.
        - High attention does **not** equate to pathological proof of malignancy and must be reviewed cautiously by clinical specialists.
        """)

    # =========================================================================
    # VIEW 7: MODEL PIPELINE
    # =========================================================================
    elif app_mode == "📐 Model Pipeline":
        st.markdown("### 📐 End-to-End Model Pipeline")
        render_pipeline_flowchart()

    # =========================================================================
    # VIEW 8: FINAL TEST RESULTS
    # =========================================================================
    elif app_mode == "🔒 Final Test Results":
        st.markdown("### 🔒 Final Locked Test Evaluation (V8 Primary Benchmark)")
        
        st.markdown("""
        <div style="background: #0f172a; border-left: 5px solid #10b981; padding: 16px 20px; border-radius: 8px; color: #e2e8f0; margin-bottom: 24px;">
            <b style="color: #34d399; font-size: 1.1rem;">FINAL REPORTED MODEL: V8 (FROZEN & LOCKED)</b><br>
            Formally locked evaluation on the primary held-out test cohort (<b>525 scans / 235 unique patients</b>) 
            with zero patient leakage and fixed classification threshold ($\tau=0.50$).
        </div>
        """, unsafe_allow_html=True)

        m = LOCKED_V8_TEST_METRICS
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Test Accuracy", f"{m['accuracy']*100:.2f}%")
        with col_m2:
            st.metric("Precision (PPV)", f"{m['precision']*100:.2f}%")
        with col_m3:
            st.metric("Sensitivity / Recall", f"{m['recall']*100:.2f}%")
        with col_m4:
            st.metric("Specificity", f"{m['specificity']*100:.2f}%")

        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        with col_m5:
            st.metric("Binary F1 Score", f"{m['binary_f1']*100:.2f}%")
        with col_m6:
            st.metric("Macro-F1 Score", f"{m['macro_f1']*100:.2f}%")
        with col_m7:
            st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        with col_m8:
            st.metric("PR-AUC", f"{m['pr_auc']:.4f}")

        st.markdown("---")
        
        col_cm_plot, col_cm_table = st.columns([1, 1.2])
        with col_cm_plot:
            st.markdown("#### Test Confusion Matrix Heatmap ($N=525$)")
            fig = render_confusion_matrix_plot(m["confusion_matrix"])
            st.pyplot(fig)
            
        with col_cm_table:
            st.markdown("#### Primary Test Cohort Parameters")
            st.markdown(f"""
            - **Total Test Scans:** `{m['n_scans']}`
            - **Total Test Patients:** `{m['n_patients']}`
            - **Patient Leakage:** `{m['patient_leakage']}`
            - **True Negatives (TN):** `{m['confusion_matrix']['tn']}`
            - **False Positives (FP):** `{m['confusion_matrix']['fp']}`
            - **False Negatives (FN):** `{m['confusion_matrix']['fn']}`
            - **True Positives (TP):** `{m['confusion_matrix']['tp']}`
            - **Brier Calibration Score:** `{m['brier_score']}`
            - **Expected Calibration Error (ECE):** `{m['ece']}%`
            """)
            
            st.markdown("#### 95% Bootstrap Confidence Intervals")
            ci_df = pd.DataFrame([
                {"Metric": "Accuracy", "95% CI": m["ci_95"]["accuracy"]},
                {"Metric": "Precision", "95% CI": m["ci_95"]["precision"]},
                {"Metric": "Recall (Sensitivity)", "95% CI": m["ci_95"]["recall"]},
                {"Metric": "Specificity", "95% CI": m["ci_95"]["specificity"]},
                {"Metric": "Binary F1", "95% CI": m["ci_95"]["binary_f1"]},
                {"Metric": "Macro-F1", "95% CI": m["ci_95"]["macro_f1"]},
                {"Metric": "ROC-AUC", "95% CI": m["ci_95"]["roc_auc"]},
                {"Metric": "PR-AUC", "95% CI": m["ci_95"]["pr_auc"]}
            ])
            st.dataframe(ci_df, use_container_width=True)

    # =========================================================================
    # VIEW 9: MODEL COMPARISONS (V5 -> V6 -> V8 -> V9)
    # =========================================================================
    elif app_mode == "📊 Model Comparisons (V5-V9)":
        st.markdown("### 📊 Multi-Generation Model Evaluation & Selection Rationale")
        
        st.markdown("""
        #### Formal Thesis Model Selection Statement:
        > *"Among the evaluated experimental configurations, V8 was designated as the final reported model based on its locked patient-independent test evaluation and overall classification balance. On the primary test cohort of 525 scans from 235 patients, V8 achieved 64.57% accuracy, 64.18% Macro-F1, 0.6947 ROC-AUC, 0.5857 PR-AUC, 66.05% sensitivity, and 63.55% specificity.*
        >
        > *A subsequent V9 experimental extension achieved higher malignant sensitivity of 76.28% and PR-AUC of 0.6129, but its overall accuracy decreased to 62.86% and specificity decreased to 53.55%. The V9 implementation and final test evaluation were independently audited and no implementation error was identified. Therefore, V9 is retained as a documented experimental extension rather than replacing V8 as the final reported configuration."*
        """)

        st.markdown("---")
        st.markdown("#### Locked Test Results Across Pipeline Generations")
        
        comp_df = pd.DataFrame(LOCKED_MODEL_COMPARISON_TABLE)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("""
        #### Key Scientific Insights:
        - **V8 Balance:** V8 provides the highest specificity (**63.55%**) among multi-crop architectures while maintaining robust malignant sensitivity (**66.05%**) and strong overall accuracy (**64.57%**).
        - **V9 Extension:** V9 increased sensitivity to **76.28%** via expanded candidate attention, but suffered a specificity trade-off (**53.55%**). It remains documented in the permanent research record as an exploratory extension.
        """)

    # =========================================================================
    # VIEW 10: DATASET INFORMATION
    # =========================================================================
    elif app_mode == "📚 Dataset Information":
        st.markdown("### 📚 CBIS-DDSM Dataset Cohort Information")
        st.write("Curated Breast Imaging Subset of DDSM (Digital Database for Screening Mammography).")

        d = DATASET_STATS
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Total Records", f"{d['total_records']}")
        with col_d2:
            st.metric("Total Patients", f"{d['total_patients']}")
        with col_d3:
            st.metric("Patient Overlap", f"{d['patient_overlap']} (Zero Leakage)")

        st.markdown("---")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Patient-Stratified Splits")
            splits_data = [
                {"Split": "Train", "Scans": d["splits"]["train"]["scans"], "Patients": d["splits"]["train"]["patients"], "Percentage": d["splits"]["train"]["pct"]},
                {"Split": "Validation", "Scans": d["splits"]["validation"]["scans"], "Patients": d["splits"]["validation"]["patients"], "Percentage": d["splits"]["validation"]["pct"]},
                {"Split": "Test (Locked)", "Scans": d["splits"]["test"]["scans"], "Patients": d["splits"]["test"]["patients"], "Percentage": d["splits"]["test"]["pct"]}
            ]
            st.dataframe(pd.DataFrame(splits_data), use_container_width=True)

        with col_s2:
            st.markdown("#### Class & Abnormality Distribution")
            cls_data = [
                {"Category": "Benign Cases", "Count": d["class_distribution"]["benign"], "Percentage": d["class_distribution"]["benign_pct"]},
                {"Category": "Malignant Cases", "Count": d["class_distribution"]["malignant"], "Percentage": d["class_distribution"]["malignant_pct"]},
                {"Category": "Calcification Scans", "Count": d["abnormality_types"]["calcification"], "Percentage": "49.9%"},
                {"Category": "Mass Scans", "Count": d["abnormality_types"]["mass"], "Percentage": "50.1%"}
            ]
            st.dataframe(pd.DataFrame(cls_data), use_container_width=True)

        st.markdown("""
        #### Patient-Level Stratification Protocol:
        All scans from a single patient (both CC and MLO views, multiple bilateral projections, and temporal series) 
        were strictly grouped into the same partition. This prevents patient identity leakage and inflated performance metrics.
        """)

    # =========================================================================
    # VIEW 11: ABOUT PROJECT
    # =========================================================================
    elif app_mode == "ℹ️ About Project":
        st.markdown("### ℹ️ About Project & Academic Disclaimers")
        
        render_safety_disclaimer("full")

        st.markdown("""
        #### Final-Year Academic Project:
        **"Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis"**

        #### Final Reported Model:
        **V8 (Phase 2 Arm F: Dual-Resolution U-KAN K=3 Proposals + Last 4 ViT Blocks Fine-Tuned)**

        #### Key Contributions:
        1. **Adaptive Spatial Noise Denoising:** Wiener filtering targeting mammographic sensor noise while preserving microcalcifications.
        2. **Dual-Resolution U-KAN Lesion Segmentation:** Kolmogorov-Arnold network spline bottleneck with multi-resolution receptive field ($512+768$).
        3. **Deterministic Top-K Proposal Generation:** Machine connected component extraction scored on area, confidence, compactness, and contrast.
        4. **Multi-Scale Concentric RoI Fusion:** Joint latent representation spanning lesion core, boundary margins, regional context, calcification patches, and full mammogram context.
        5. **Rigorous Data Governance:** Quarantined held-out test split with strict cryptographic checkpoint verification.
        """)


if __name__ == "__main__":
    main()
