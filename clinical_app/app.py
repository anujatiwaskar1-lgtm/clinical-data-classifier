"""
app.py — Clinical Data Classifier
Streamlit web app — deploy free on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from classifier import load_and_train, structured_classify

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Clinical Data Classifier",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
CAT_COLORS = {
    "Demographics":       "#7F77DD",
    "Vital Signs":        "#1D9E75",
    "Laboratory Reports": "#378ADD",
    "Pathology Reports":  "#EF9F27",
    "Microbiology Data":  "#D85A30",
    "Clinical Notes":     "#639922",
}
CAT_ICONS = {
    "Demographics":       "👤",
    "Vital Signs":        "💓",
    "Laboratory Reports": "🧪",
    "Pathology Reports":  "🔬",
    "Microbiology Data":  "🦠",
    "Clinical Notes":     "📋",
}


# ─────────────────────────────────────────────────────────
#  LOAD MODEL (cached — only trains once)
# ─────────────────────────────────────────────────────────

import pickle

@st.cache_resource(show_spinner=False)
def get_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("le.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=72)
    st.title("Clinical Classifier")
    st.caption("Classify and extract structured data from clinical text.")
    st.divider()

    
    st.divider()
    st.subheader("Settings")
    threshold = st.slider(
        "Confidence threshold (%)",
        min_value=10, max_value=50, value=15,
        help="Minimum confidence required to show a category."
    )
    st.caption("Lower = more categories detected. Higher = only strong matches.")
    st.divider()
    st.caption("Built with Scikit-learn + Streamlit")
    st.caption("Dataset: Kaggle Medical Transcriptions")

# ─────────────────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🏥 Clinical Data Classifier</h1>"
    "<p style='color:#888;margin-top:4px'>Detects all clinical categories in your text and extracts structured field values.</p>",
    unsafe_allow_html=True
)
st.divider()

# Load model with progress

with st.spinner("Loading model..."):
    try:
        model, le = get_model()
        st.success("Model ready.", icon="✅")
    except FileNotFoundError:
        st.error("model.pkl or le.pkl not found. Please add them to the folder.")
        st.stop()

# ── Text input ────────────────────────────────────────────
st.subheader("Enter Clinical Text")
user_text = st.text_area(
    label     = "Paste any clinical record, note, or lab report:",
    value     = "",
    height    = 130,
    placeholder = (
        "e.g. Anuja Sharma, 28 year old female, BP 110/70 mmHg, "
        "complaints of fever for 3 days, paracetamol started."
    ),
)

classify_btn = st.button("🔍  Classify", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────
if classify_btn:
    if not user_text.strip():
        st.warning("Please enter some clinical text first.")
        st.stop()

    with st.spinner("Classifying..."):
        output = structured_classify(user_text, model, le)

    if not output:
        st.warning(
            "No categories detected above the threshold. "
            "Try lowering the confidence threshold in the sidebar, "
            "or enter more detailed clinical text."
        )
        st.stop()

    st.divider()
    n = len(output)
    st.subheader(f"Results — {n} categor{'y' if n == 1 else 'ies'} detected")

    # ── Category cards ────────────────────────────────────
    cols = st.columns(n)
    for col, (cat, info) in zip(cols, output.items()):
        icon  = CAT_ICONS.get(cat, "📁")
        color = CAT_COLORS.get(cat, "#888")
        with col:
            st.markdown(
                f"""
                <div style='background:{color}18;border:2px solid {color};
                border-radius:14px;padding:18px 10px;text-align:center;'>
                  <div style='font-size:32px;line-height:1.2'>{icon}</div>
                  <div style='font-weight:700;font-size:12px;color:{color};
                  margin:6px 0 2px;letter-spacing:0.03em'>{cat}</div>
                  <div style='font-size:26px;font-weight:800;color:{color}'>
                  {info['confidence']:.0f}%</div>
                  <div style='font-size:10px;color:#aaa'>confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Two-column layout: table + chart ─────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Extracted Fields")
        rows = []
        for cat, info in output.items():
            for field, value in info["entities"].items():
                rows.append({
                    "Category":   cat,
                    "Field":      field,
                    "Value":      value,
                    "Confidence": f"{info['confidence']:.0f}%",
                })

        if rows:
            df_out = pd.DataFrame(rows)

            def style_row(row):
                color = CAT_COLORS.get(row["Category"], "#888")
                bg    = color + "18"
                return [f"background-color:{bg}"] * len(row)

            st.dataframe(
                df_out.style.apply(style_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            csv = df_out.to_csv(index=False)
            st.download_button(
                "⬇️  Download as CSV",
                data     = csv,
                file_name = "clinical_extraction.csv",
                mime     = "text/csv",
                use_container_width = True,
            )
        else:
            st.info(
                "Categories were detected but no specific field values could be extracted. "
                "Try adding more specific clinical details to your text."
            )

    with right:
        st.subheader("Confidence Chart")
        chart_df = pd.DataFrame([
            {"Category": cat, "Confidence": info["confidence"]}
            for cat, info in output.items()
        ]).sort_values("Confidence", ascending=True)

        colors = [CAT_COLORS.get(c, "#888") for c in chart_df["Category"]]

        fig = go.Figure(go.Bar(
            x           = chart_df["Confidence"],
            y           = chart_df["Category"],
            orientation = "h",
            marker_color = colors,
            text        = [f"{v:.1f}%" for v in chart_df["Confidence"]],
            textposition = "outside",
        ))
        fig.update_layout(
            margin       = dict(l=0, r=60, t=10, b=10),
            xaxis        = dict(range=[0, 115], showgrid=False, zeroline=False,
                                showticklabels=False),
            yaxis        = dict(showgrid=False),
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor= "rgba(0,0,0,0)",
            height       = max(220, n * 58),
            showlegend   = False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-category detail expanders ─────────────────────
    st.divider()
    st.subheader("Category Details")
    for cat, info in output.items():
        icon  = CAT_ICONS.get(cat, "📁")
        color = CAT_COLORS.get(cat, "#888")
        with st.expander(f"{icon}  {cat}  —  {info['confidence']:.0f}% confidence"):
            if info["entities"]:
                for field, value in info["entities"].items():
                    c1, c2 = st.columns([1, 2])
                    c1.markdown(f"**{field}**")
                    c2.markdown(
                        f"<span style='background:{color}22;padding:2px 10px;"
                        f"border-radius:6px;font-weight:600'>{value}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No specific field values extracted for this category.")
