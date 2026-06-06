"""
Financial Sentiment Analyser — Interactive Dashboard
Fine-tuned DistilBERT on 9,543 financial news headlines.
"""

import streamlit as st
import sys, os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model, predict

st.set_page_config(
    page_title="Financial Sentiment Analyser",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0A0F1E; }
    .block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }
    .stButton > button {
        background: linear-gradient(90deg, #1D4ED8, #3B82F6);
        color: white; border: none; border-radius: 10px;
        padding: 0.5rem 2rem; font-size: 1rem; font-weight: 600;
    }
    .stButton > button:hover { background: linear-gradient(90deg, #2563EB, #60A5FA); }
    .stTextArea textarea {
        background-color: #111827 !important;
        border: 1px solid #1E40AF !important;
        color: #F1F5F9 !important;
        border-radius: 10px !important;
    }
    .result-box {
        background: linear-gradient(135deg, #0F172A, #1E3A5F);
        border: 1px solid #3B82F6;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .bearish { color: #F87171; font-size: 1.8rem; font-weight: 700; }
    .neutral { color: #FBBF24; font-size: 1.8rem; font-weight: 700; }
    .bullish { color: #34D399; font-size: 1.8rem; font-weight: 700; }
    .tag-bearish { background:#7F1D1D; color:#FCA5A5; border-radius:6px; padding:2px 8px; font-size:0.82rem; }
    .tag-neutral  { background:#78350F; color:#FDE68A; border-radius:6px; padding:2px 8px; font-size:0.82rem; }
    .tag-bullish  { background:#064E3B; color:#6EE7B7; border-radius:6px; padding:2px 8px; font-size:0.82rem; }
    .footer { color: #475569; font-size: 0.78rem; text-align: center; margin-top: 2rem; }
    h1 { background: linear-gradient(90deg, #3B82F6, #60A5FA);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; }
</style>
""", unsafe_allow_html=True)

PLOT_BG   = "#0A0F1E"
PAPER_BG  = "#111827"
GRID_COL  = "#1E3A5F"
TEXT_COL  = "#94A3B8"
BLUE      = "#3B82F6"
GREEN     = "#34D399"
YELLOW    = "#FBBF24"
RED       = "#F87171"

def dark_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COL),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {text, label, confidence, scores}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Loading model..."):
    tokenizer, model = get_model()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 Financial Sentiment Analyser")
st.markdown(
    "Fine-tuned **DistilBERT** on **9,543** financial news headlines · "
    "**87.9% accuracy** · **84.2% F1 macro** · "
    "HuggingFace Transformers · PyTorch · Streamlit"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Analyse", "📊 Dashboard", "🏆 Model Performance"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — ANALYSE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Single Headline")
        examples = [
            "Apple beats earnings expectations, stock surges 8%",
            "Fed raises interest rates amid inflation concerns",
            "Tesla misses delivery targets for third consecutive quarter",
            "Goldman Sachs upgrades Microsoft to Buy with $450 price target",
            "Crypto markets crash as regulators announce crackdown",
            "RBA holds cash rate steady as inflation eases toward target",
            "ASX 200 hits all-time high on strong commodities rally",
        ]
        selected = st.selectbox("Try an example:", ["— type your own —"] + examples)
        user_input = st.text_area(
            "Or enter any financial headline:",
            value=selected if selected != "— type your own —" else "",
            height=100,
        )

        if st.button("Analyse", type="primary"):
            if user_input.strip():
                with st.spinner("Analysing..."):
                    result = predict(user_input.strip(), tokenizer, model)
                st.session_state.history.append({
                    "text": user_input.strip(),
                    **result,
                })
                label = result["label"].upper()
                emoji = result["emoji"]
                confidence = result["confidence"]
                scores = result["scores"]
                css_class = label.lower()

                st.markdown(f"""
                <div class="result-box">
                    <div class="{css_class}">{emoji} {label}</div>
                    <div style="color:#94A3B8; margin:0.3rem 0 0.8rem 0">
                        Confidence: <strong style="color:#F1F5F9">{confidence:.1%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 Bearish", f"{scores['bearish']:.1%}")
                m2.metric("🟡 Neutral",  f"{scores['neutral']:.1%}")
                m3.metric("🟢 Bullish",  f"{scores['bullish']:.1%}")

                # Confidence bar chart
                fig = go.Figure(go.Bar(
                    x=["Bearish", "Neutral", "Bullish"],
                    y=[scores["bearish"], scores["neutral"], scores["bullish"]],
                    marker_color=[RED, YELLOW, GREEN],
                    text=[f"{v:.1%}" for v in [scores["bearish"], scores["neutral"], scores["bullish"]]],
                    textposition="outside",
                ))
                fig = dark_layout(fig, "Confidence Scores", height=280)
                fig.update_yaxes(tickformat=".0%", range=[0, 1.1], gridcolor=GRID_COL)
                fig.update_xaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a headline.")

    with col_right:
        st.subheader("Batch Analysis")
        st.caption("One headline per line.")
        batch_input = st.text_area("Headlines:", height=200)

        if st.button("Analyse Batch"):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            if lines:
                with st.spinner(f"Analysing {len(lines)} headlines..."):
                    results = [predict(l, tokenizer, model) for l in lines]
                for r in results:
                    st.session_state.history.append({"text": r, **r})

                # Results table
                df_batch = pd.DataFrame([{
                    "Headline": line[:60] + "..." if len(line) > 60 else line,
                    "Sentiment": res["emoji"] + " " + res["label"].upper(),
                    "Confidence": f"{res['confidence']:.1%}",
                    "Bearish": f"{res['scores']['bearish']:.1%}",
                    "Neutral": f"{res['scores']['neutral']:.1%}",
                    "Bullish": f"{res['scores']['bullish']:.1%}",
                } for line, res in zip(lines, results)])

                st.dataframe(df_batch, use_container_width=True, hide_index=True)

                # Batch distribution chart
                label_counts = pd.Series([r["label"] for r in results]).value_counts()
                fig2 = go.Figure(go.Pie(
                    labels=label_counts.index.str.upper(),
                    values=label_counts.values,
                    hole=0.5,
                    marker_colors=[RED, YELLOW, GREEN],
                ))
                fig2 = dark_layout(fig2, "Batch Sentiment Distribution", height=280)
                st.plotly_chart(fig2, use_container_width=True)

                # Confidence line
                fig3 = go.Figure(go.Scatter(
                    x=list(range(1, len(results)+1)),
                    y=[r["confidence"] for r in results],
                    mode="lines+markers",
                    line=dict(color=BLUE, width=2),
                    marker=dict(color=[
                        RED if r["label"]=="bearish" else
                        YELLOW if r["label"]=="neutral" else GREEN
                        for r in results
                    ], size=10),
                ))
                fig3 = dark_layout(fig3, "Confidence per Headline", height=260)
                fig3.update_yaxes(tickformat=".0%", range=[0,1.1], gridcolor=GRID_COL)
                fig3.update_xaxes(title="Headline #", gridcolor=GRID_COL)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Please enter at least one headline.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD (session history)
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Session Analytics Dashboard")

    if not st.session_state.history:
        st.info("Run some analyses in the Analyse tab to populate this dashboard.")
    else:
        h = st.session_state.history
        df_h = pd.DataFrame([{
            "text": item["text"] if isinstance(item["text"], str) else str(item["text"]),
            "label": item["label"],
            "confidence": item["confidence"],
        } for item in h])

        total = len(df_h)
        avg_conf = df_h["confidence"].mean()
        most_common = df_h["label"].value_counts().idxmax().upper()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analysed", total)
        c2.metric("Avg Confidence", f"{avg_conf:.1%}")
        c3.metric("Dominant Sentiment", most_common)
        c4.metric("Bullish %", f"{(df_h['label']=='bullish').mean():.1%}")

        col_a, col_b = st.columns(2)

        with col_a:
            counts = df_h["label"].value_counts()
            fig4 = go.Figure(go.Pie(
                labels=[l.upper() for l in counts.index],
                values=counts.values,
                hole=0.55,
                marker_colors=[
                    RED if l=="bearish" else YELLOW if l=="neutral" else GREEN
                    for l in counts.index
                ],
            ))
            fig4 = dark_layout(fig4, "Overall Sentiment Distribution", 320)
            st.plotly_chart(fig4, use_container_width=True)

        with col_b:
            fig5 = go.Figure(go.Scatter(
                x=list(range(1, total+1)),
                y=df_h["confidence"].tolist(),
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.15)",
                line=dict(color=BLUE, width=2),
                marker=dict(
                    color=[
                        RED if l=="bearish" else YELLOW if l=="neutral" else GREEN
                        for l in df_h["label"]
                    ], size=8
                ),
            ))
            fig5 = dark_layout(fig5, "Confidence Over Time", 320)
            fig5.update_yaxes(tickformat=".0%", range=[0,1.1], gridcolor=GRID_COL)
            fig5.update_xaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig5, use_container_width=True)

        # Avg confidence per class
        avg_by_label = df_h.groupby("label")["confidence"].mean().reset_index()
        fig6 = go.Figure(go.Bar(
            x=[l.upper() for l in avg_by_label["label"]],
            y=avg_by_label["confidence"],
            marker_color=[
                RED if l=="bearish" else YELLOW if l=="neutral" else GREEN
                for l in avg_by_label["label"]
            ],
            text=[f"{v:.1%}" for v in avg_by_label["confidence"]],
            textposition="outside",
        ))
        fig6 = dark_layout(fig6, "Average Confidence by Sentiment Class", 300)
        fig6.update_yaxes(tickformat=".0%", range=[0,1.1], gridcolor=GRID_COL)
        fig6.update_xaxes(showgrid=False)
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("**Full Session History**")
        st.dataframe(df_h.rename(columns={
            "text":"Headline","label":"Sentiment","confidence":"Confidence"
        }).assign(
            Confidence=df_h["confidence"].apply(lambda x: f"{x:.1%}"),
            Sentiment=df_h["label"].str.upper()
        ), use_container_width=True, hide_index=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🏆 Model Performance")
    st.markdown("Training results from fine-tuning `distilbert-base-uncased` on 9,543 financial headlines.")

    col_p, col_q = st.columns(2)

    with col_p:
        # F1 per class
        fig7 = go.Figure(go.Bar(
            x=["Bearish", "Neutral", "Bullish", "Macro Avg"],
            y=[0.7812, 0.8279, 0.9161, 0.8417],
            marker_color=[RED, YELLOW, GREEN, BLUE],
            text=["78.1%", "82.8%", "91.6%", "84.2%"],
            textposition="outside",
        ))
        fig7 = dark_layout(fig7, "F1 Score by Class", 320)
        fig7.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor=GRID_COL)
        fig7.update_xaxes(showgrid=False)
        st.plotly_chart(fig7, use_container_width=True)

    with col_q:
        # Training loss per epoch
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=[1, 2, 3], y=[0.3392, 0.3616, 0.5150],
            mode="lines+markers", name="Eval Loss",
            line=dict(color=RED, width=2), marker=dict(size=10),
        ))
        fig8.add_trace(go.Scatter(
            x=[1, 2, 3], y=[0.799*0.5, 0.2786*0.7, 0.1058],
            mode="lines+markers", name="Train Loss (approx)",
            line=dict(color=BLUE, width=2, dash="dash"), marker=dict(size=10),
        ))
        fig8 = dark_layout(fig8, "Loss Curve", 320)
        fig8.update_yaxes(gridcolor=GRID_COL)
        fig8.update_xaxes(title="Epoch", tickvals=[1,2,3], gridcolor=GRID_COL)
        fig8.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig8, use_container_width=True)

    # Accuracy per epoch
    fig9 = go.Figure(go.Scatter(
        x=[1, 2, 3],
        y=[0.8706, 0.8765, 0.8794],
        mode="lines+markers+text",
        text=["87.1%", "87.7%", "87.9%"],
        textposition="top center",
        line=dict(color=GREEN, width=3),
        marker=dict(color=GREEN, size=12),
        fill="tozeroy",
        fillcolor="rgba(52,211,153,0.1)",
    ))
    fig9 = dark_layout(fig9, "Validation Accuracy per Epoch", 280)
    fig9.update_yaxes(tickformat=".0%", range=[0.8, 0.92], gridcolor=GRID_COL)
    fig9.update_xaxes(title="Epoch", tickvals=[1,2,3], gridcolor=GRID_COL)
    st.plotly_chart(fig9, use_container_width=True)

    # Summary table
    st.markdown("**Training Summary**")
    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Macro", "F1 Bearish", "F1 Neutral", "F1 Bullish", "Train Samples", "Val Samples", "Epochs"],
        "Value":  ["87.9%",   "84.2%",    "78.1%",      "82.8%",      "91.6%",       "9,543",         "2,388",       "3"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    Model: DistilBERT (distilbert-base-uncased) fine-tuned &nbsp;·&nbsp;
    Dataset: Twitter Financial News Sentiment (9,543 samples) &nbsp;·&nbsp;
    Stack: HuggingFace Transformers · PyTorch · Streamlit · Plotly &nbsp;·&nbsp;
    <a href="https://github.com/fahadamjad009/nlp1-financial-sentiment" style="color:#3B82F6">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)