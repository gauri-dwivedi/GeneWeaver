"""
GeneWeaver: UI Components and Styling
=====================================
Contains custom styling, device status badges, color-coded sequence formatters,
and reusable card components for the Streamlit application.
"""

import streamlit as st
import torch
from src.gpu_engine import GPUEngine


# Custom Deep Navy & Bio-Tech Theme CSS
CUSTOM_CSS = """
<style>
    /* Global Typography & Font settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* GeneWeaver Header Hero */
    .hero-banner {
        background: linear-gradient(135deg, #0A192F 0%, #172A45 50%, #1E3E62 100%);
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.75rem;
        color: #F1F5F9;
        font-weight: 700;
        margin: 4px 0 0 0;
    }

    /* Device Badges */
    .device-badge-gpu {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .device-badge-cpu {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Nucleotide Sequence Viewer */
    .seq-box {
        background-color: #0B132B;
        border: 1px solid #1C2541;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1.5px;
        word-break: break-all;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    .base-a { color: #4ADE80; font-weight: 600; } /* Green */
    .base-c { color: #60A5FA; font-weight: 600; } /* Blue */
    .base-g { color: #FBBF24; font-weight: 600; } /* Orange */
    .base-t { color: #F87171; font-weight: 600; } /* Red */
    .base-other { color: #A78BFA; }

    /* Alignment Visual Highlighting */
    .align-view-container {
        background: #090D16;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .match-char { color: #10B981; font-weight: bold; }
    .mismatch-char { color: #EF4444; font-weight: bold; background: rgba(239, 68, 68, 0.2); border-radius: 2px; }
    .pam-char { color: #F59E0B; font-weight: bold; text-decoration: underline; }
</style>
"""


def apply_custom_theme():
    """Inject global CSS rules into the active Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, icon: str = "🧬"):
    """
    Render a clean, modern header banner with icon, title, and description.
    """
    html = f"""
    <div class="hero-banner">
        <h1 class="hero-title">{icon} {title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_device_status_sidebar():
    """
    Render the active hardware detection badge in the Streamlit sidebar.
    """
    gpu_engine = GPUEngine()
    device_info = gpu_engine.get_device_info()

    st.sidebar.markdown("### 🖥️ Hardware Acceleration")
    if device_info["cuda_available"]:
        st.sidebar.markdown(
            f'<div class="device-badge-gpu">🚀 CUDA GPU Active: {device_info.get("gpu_name", "GPU")}</div>',
            unsafe_allow_html=True
        )
        st.sidebar.caption("PyTorch is accelerating sequence batch matching using GPU CUDA cores.")
    else:
        st.sidebar.markdown(
            '<div class="device-badge-cpu">⚡ Processing Device: CPU (Fallback)</div>',
            unsafe_allow_html=True
        )
        st.sidebar.caption(
            "CUDA-capable GPU not detected or driver unavailable. "
            "Running CPU reference engine and PyTorch CPU vectorized fallback."
        )

    st.sidebar.markdown("---")


def format_sequence_html(seq: str, max_display_chars: int = 1000) -> str:
    """
    Generate syntax-highlighted HTML for a DNA sequence:
    Adenine (A) = Green, Cytosine (C) = Blue, Guanine (G) = Yellow, Thymine (T) = Red.
    """
    display_seq = seq[:max_display_chars]
    colored_bases = []
    for ch in display_seq:
        if ch == "A":
            colored_bases.append('<span class="base-a">A</span>')
        elif ch == "C":
            colored_bases.append('<span class="base-c">C</span>')
        elif ch == "G":
            colored_bases.append('<span class="base-g">G</span>')
        elif ch == "T":
            colored_bases.append('<span class="base-t">T</span>')
        else:
            colored_bases.append(f'<span class="base-other">{ch}</span>')

    rendered = "".join(colored_bases)
    if len(seq) > max_display_chars:
        rendered += f' <span style="color:#94A3B8;">... (+{len(seq) - max_display_chars:,} more bp)</span>'

    return f'<div class="seq-box">{rendered}</div>'


def render_visual_alignment(guide: str, target: str, pam: str, mismatches: list[int]):
    """
    Render a clean graphical 3-line ASCII/colored alignment view:
    Line 1: 5' Guide sequence 3'
    Line 2: Match connector (| = match, . = mismatch)
    Line 3: 5' Genomic target + PAM 3'
    """
    guide_html = []
    bar_html = []
    target_html = []

    for idx, (g, t) in enumerate(zip(guide, target)):
        pos = idx + 1
        if pos in mismatches:
            guide_html.append(f'<span class="mismatch-char">{g}</span>')
            bar_html.append('<span class="mismatch-char">.</span>')
            target_html.append(f'<span class="mismatch-char">{t}</span>')
        else:
            guide_html.append(f'<span class="match-char">{g}</span>')
            bar_html.append('<span class="match-char">|</span>')
            target_html.append(f'<span class="match-char">{t}</span>')

    # Add PAM to target line
    pam_formatted = f'<span class="pam-char">{pam}</span>'

    alignment_block = f"""
    <div class="align-view-container">
        <div><strong>Guide  (5'-3'): </strong> {''.join(guide_html)}</div>
        <div><strong>Match:        </strong> {''.join(bar_html)}</div>
        <div><strong>Target (5'-3'): </strong> {''.join(target_html)} {pam_formatted} <span style="color:#F59E0B; font-size:0.8rem;">[PAM]</span></div>
    </div>
    """
    st.markdown(alignment_block, unsafe_allow_html=True)

