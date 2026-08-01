import base64
import threading
import time
import requests
import streamlit as st
import os
import pandas as pd
import altair as alt
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

st.set_page_config(
    page_title="Oil Tank Volume Estimator",
    layout="wide",
    page_icon="O",
    initial_sidebar_state="expanded",
)

# ──────────────────────── Theme ─────────────────────────────────────

st.markdown(
    """
<style>
    :root {
        --blue-900: #0f2d52;
        --blue-700: #1e4f8a;
        --blue-500: #2b6cb0;
        --blue-100: #dbeafe;
        --blue-50:  #f0f6fc;
        --gray-900: #1a1f2e;
        --gray-700: #374151;
        --gray-600: #4b5563;
        --gray-500: #6b7280;
        --gray-400: #9ca3af;
        --gray-300: #cbd5e1;
        --gray-200: #e2e8f0;
        --gray-100: #f1f5f9;
        --white:    #ffffff;
        --border:   #94a3b8;
        --border-strong: #64748b;
        --green:    #16a34a;
        --green-dark: #15803d;
    }

    /* Force full white app shell */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main,
    .block-container,
    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: #ffffff !important;
        color: var(--gray-900) !important;
    }

    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid var(--border);
    }

    .block-container {
        padding-top: 5rem;
        max-width: 1200px;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--gray-700) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--blue-900) !important;
    }

    /* Widget labels — darker & readable */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {
        color: var(--gray-700) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }

    .stMarkdown p,
    .stCaption {
        color: var(--gray-600) !important;
    }

    /* Buttons — outlined, not solid blue blocks */
    .stButton > button {
        background-color: #ffffff !important;
        color: var(--gray-900) !important;
        border: 1.5px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .stButton > button:hover {
        background-color: var(--gray-100) !important;
        border-color: var(--blue-700) !important;
        color: var(--blue-700) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1.5rem 1.25rem !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--border-strong) !important;
        background-color: var(--gray-100) !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: var(--gray-900) !important;
        border: 1.5px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: var(--blue-50) !important;
        border-color: var(--blue-700) !important;
        color: var(--blue-700) !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: var(--gray-600) !important;
    }

    /* Selectbox — visible bordered box */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--gray-900) !important;
        min-height: 38px !important;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
    [data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--blue-700) !important;
        box-shadow: 0 0 0 2px rgba(30, 79, 138, 0.15) !important;
    }

    /* Number inputs — clear visible field */
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] > div > div {
        background-color: #ffffff !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 6px !important;
    }

    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        border: none !important;
        color: var(--gray-900) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: var(--gray-900) !important;
    }

    [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border-color: var(--blue-700) !important;
        box-shadow: 0 0 0 2px rgba(30, 79, 138, 0.15) !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: var(--gray-100) !important;
        border-left: 1.5px solid var(--border) !important;
        color: var(--gray-700) !important;
    }

    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--blue-700) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 1px var(--border) !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stThumbValue"] {
        color: var(--blue-700) !important;
        font-weight: 600 !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: var(--gray-300) !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background-color: var(--blue-700) !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0 !important;
    }

    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: var(--gray-500) !important;
        font-size: 0.82rem !important;
    }

    /* Tooltip / help icons */
    [data-testid="stTooltipIcon"] {
        color: var(--gray-500) !important;
    }

    .stApp {
        color-scheme: light !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #ffffff !important;
        color: var(--gray-900) !important;
    }

    [data-testid="stAlert"] {
        background-color: #ffffff !important;
        border: 1.5px solid var(--border) !important;
    }

    .page-header {
        margin-bottom: 0.25rem;
        color: var(--blue-900);
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        color: var(--gray-600);
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin: 1.5rem 0 1.75rem;
    }

    @media (max-width: 1100px) {
        .insight-grid { grid-template-columns: repeat(3, 1fr); }
    }

    @media (max-width: 700px) {
        .insight-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .insight-card {
        background: var(--white);
        border: 1.5px solid var(--border);
        border-top: 3px solid var(--blue-700);
        border-radius: 8px;
        padding: 18px 16px;
        min-height: 108px;
        box-shadow: 0 1px 2px rgba(15, 45, 82, 0.05);
    }

    .insight-label {
        color: var(--gray-600);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .insight-value {
        color: var(--blue-900);
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 4px;
    }

    .insight-meta {
        color: var(--gray-400);
        font-size: 0.78rem;
    }

    .section-title {
        color: var(--blue-900);
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1.5px solid var(--border);
    }

    .legend-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid var(--gray-200);
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--gray-600);
        font-size: 0.82rem;
    }

    .legend-swatch {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        border: 1px solid var(--gray-200);
        background: var(--blue-500);
    }

    .legend-swatch.muted { background: #94a3b8; }
    .legend-swatch.light { background: #cbd5e1; }

    .analysis-panel {
        background: var(--white);
        border: 1.5px solid var(--border);
        border-radius: 8px;
        padding: 20px 22px;
        margin: 1rem 0 1.5rem;
    }

    .progress-label {
        color: var(--gray-900);
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .win-progress-track {
        width: 100%;
        height: 22px;
        background: #e5e7eb;
        border: 1px solid #cbd5e1;
        border-radius: 2px;
        overflow: hidden;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.06);
    }

    .win-progress-fill {
        height: 100%;
        background: linear-gradient(180deg, #22c55e 0%, #16a34a 55%, #15803d 100%);
        transition: width 0.18s ease-out;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    .detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }

    .detail-table th {
        text-align: left;
        color: var(--gray-600);
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 10px 12px;
        border-bottom: 1.5px solid var(--border);
        background: var(--gray-100);
    }

    .detail-table td {
        padding: 11px 12px;
        border-bottom: 1px solid var(--gray-300);
        color: var(--gray-900);
    }

    .detail-table tr:last-child td {
        border-bottom: none;
    }

    .empty-state {
        color: var(--gray-600);
        font-size: 0.9rem;
        padding: 14px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────── Helpers ───────────────────────────────────

def render_insight_card(label, value, meta=""):
    meta_html = f'<div class="insight-meta">{meta}</div>' if meta else ""
    return f"""<div class="insight-card">
<div class="insight-label">{label}</div>
<div class="insight-value">{value}</div>
{meta_html}
</div>"""


def render_detail_table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
    <table class="detail-table">
        <thead><tr>{head}</tr></thead>
        <tbody>{body}</tbody>
    </table>
    """


def process(image, server_url: str, conf: float = 0.15):
    multipart = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    return requests.post(
        f"{server_url}?conf={conf}",
        data=multipart,
        headers={"content-type": multipart.content_type, "accept": "application/json"},
        timeout=8000,
    )


def n_barrels_calculator(height, diameter):
    radius = diameter / 2
    return radius * radius * height * 3.14 * 6.28


def run_analysis_with_progress(uploaded_file, conf_threshold, backend_url):
    progress_box = st.empty()
    result_holder = {"response": None, "error": None}

    def api_call():
        try:
            uploaded_file.seek(0)
            result_holder["response"] = process(
                uploaded_file,
                backend_url,
                conf=conf_threshold,
            )
        except Exception as exc:
            result_holder["error"] = str(exc)

    worker = threading.Thread(target=api_call, daemon=True)
    worker.start()

    stages = [
        (12, "Initializing analysis engine"),
        (28, "Uploading satellite image"),
        (48, "Running tank detection model"),
        (68, "Extracting shadow regions"),
        (86, "Estimating fill volumes"),
        (96, "Preparing analysis report"),
    ]

    progress = 0
    stage_idx = 0

    while worker.is_alive():
        target, label = stages[min(stage_idx, len(stages) - 1)]
        progress = min(progress + 2, target)
        progress_box.markdown(
            f"""
            <div class="analysis-panel">
                <div class="progress-label">{label}...</div>
                <div class="win-progress-track">
                    <div class="win-progress-fill" style="width:{progress}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if progress >= target and stage_idx < len(stages) - 1:
            stage_idx += 1
        time.sleep(0.12)

    worker.join()

    progress_box.markdown(
        """
        <div class="analysis-panel">
            <div class="progress-label">Analysis complete</div>
            <div class="win-progress-track">
                <div class="win-progress-fill" style="width:100%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(0.35)
    progress_box.empty()

    if result_holder["error"]:
        raise RuntimeError(result_holder["error"])

    return result_holder["response"]


# ──────────────────────── Sidebar ─────────────────────────────────────

st.sidebar.markdown("### Oil Tank Estimator")
st.sidebar.markdown(
    "Detect floating-head oil tanks from satellite imagery and estimate fill volumes."
)
st.sidebar.image(
    "assets/tank.jpg",
    use_container_width=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Detection settings**")
conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.15,
    step=0.05,
    help="Lower values detect more tanks. Higher values are stricter.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tank dimensions**")
st.sidebar.caption("Enter average floating-head tank size in meters.")
diameter = st.sidebar.number_input(
    "Average FHT diameter (m)",
    value=8.0,
    format="%.2f",
    min_value=0.1,
    help="Typical diameter range: 5–20 m",
)
height = st.sidebar.number_input(
    "Average FHT height (m)",
    value=25.0,
    format="%.2f",
    min_value=0.1,
    help="Typical height range: 10–30 m",
)

# ──────────────────────── Main Page ───────────────────────────────────

st.markdown('<div class="page-header">Tank Volume Estimator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Upload a satellite image to detect storage tanks and estimate floating-head tank volumes.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title" style="margin-top:0">Upload satellite image</div>',
    unsafe_allow_html=True,
)
st.caption("Supported formats: JPG, PNG — max 200 MB per file.")
uploaded_file = st.file_uploader(
    "Drag and drop file here, or click Browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="High-resolution image of an oil refinery or tank farm.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000/prediction/")

    try:
        segments = run_analysis_with_progress(uploaded_file, conf_threshold, backend_url)
    except RuntimeError as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

    if segments.status_code != 200:
        st.error(f"API error {segments.status_code}: {segments.text}")
    else:
        data = segments.json()
        img_bytes = base64.b64decode(data["image_encoded"])

        fht_results = data["results"][0] if data["results"] else []
        all_dets = data.get("all_detections", [])

        n_fht = len(fht_results)
        n_frt = len([d for d in all_dets if d.get("class_id") == 1])
        n_tc = len([d for d in all_dets if d.get("class_id") == 2])
        n_total = len(all_dets)

        n_barrels = n_barrels_calculator(height, diameter)
        total_barrels = sum(
            float(r["volumes"]) * n_barrels
            for r in fht_results
            if r["volumes"] != "N/A"
        )
        avg_fill = (
            sum(float(r["volumes"]) for r in fht_results if r["volumes"] != "N/A") / n_fht
            if n_fht
            else 0.0
        )

        st.markdown(
            f"""<div class="insight-grid">
{render_insight_card("Total detected", n_total, "All tank classes")}
{render_insight_card("Floating head", n_fht, "Volume estimated")}
{render_insight_card("Fixed roof", n_frt, "Detection only")}
{render_insight_card("Tank clusters", n_tc, "Grouped structures")}
{render_insight_card("Estimated barrels", f"{total_barrels:,.0f}", f"Avg fill {avg_fill:.1%}")}
</div>""",
            unsafe_allow_html=True,
        )

        col_result, col_original = st.columns([1.2, 0.8], gap="large")

        with col_result:
            st.markdown('<div class="section-title">Detection results</div>', unsafe_allow_html=True)
            st.image(img_bytes, use_container_width=True)

        with col_original:
            st.markdown('<div class="section-title">Original image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        st.markdown('<div class="section-title">Detection details</div>', unsafe_allow_html=True)
        if fht_results:
            chart_data = []
            for i, det in enumerate(fht_results, start=1):
                vol = det["volumes"]
                barrels = float(vol) * n_barrels if vol != "N/A" else 0
                chart_data.append({
                    "Tank ID": f"FHT {i}",
                    "Confidence": f"{float(det['confidence']):.1%}",
                    "Fill Level": f"{float(vol):.1%}" if vol != "N/A" else "N/A",
                    "Est. Barrels": f"{barrels:,.0f}"
                })

            df = pd.DataFrame(chart_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div class="empty-state">No floating-head tanks detected in this image.</div>',
                unsafe_allow_html=True,
            )

        frt_dets = [d for d in all_dets if d.get("class_id") == 1]
        if frt_dets:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Fixed roof tanks**", unsafe_allow_html=True)
            frt_rows = [
                [f"FRT {i}", f"{float(det['confidence']):.1%}", "Not applicable"]
                for i, det in enumerate(frt_dets, start=1)
            ]
            st.markdown(
                render_detail_table(["Tank ID", "Confidence", "Volume estimate"], frt_rows),
                unsafe_allow_html=True,
            )

        tc_dets = [d for d in all_dets if d.get("class_id") == 2]
        if tc_dets:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Tank clusters**", unsafe_allow_html=True)
            tc_rows = [
                [f"TC {i}", f"{float(det['confidence']):.1%}", "Grouped detection"]
                for i, det in enumerate(tc_dets, start=1)
            ]
            st.markdown(
                render_detail_table(["Cluster ID", "Confidence", "Notes"], tc_rows),
                unsafe_allow_html=True,
            )
