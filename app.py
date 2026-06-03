import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Hit Predictor · Spotify",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme / CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark background */
  .stApp { background-color: #1E1E2E; color: #E8E8F0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #16162A;
    border-right: 1px solid #2A2A4A;
  }
  [data-testid="stSidebar"] * { color: #E8E8F0 !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #2A2A3E;
    border: 1px solid #3A3A5A;
    border-radius: 12px;
    padding: 28px 32px;
  }
  [data-testid="stMetricLabel"] { color: #9999BB !important; font-size: 15px !important; }
  [data-testid="stMetricValue"] { color: #FF4D8B !important; font-size: 36px !important; font-weight: 700 !important; }

  /* Sliders — more vertical breathing room */
  [data-testid="stSlider"] { padding-top: 12px !important; padding-bottom: 12px !important; }
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: #FF4D8B !important; }

  /* Remove gap between expander and the section below it */
  [data-testid="stExpander"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
  [data-testid="stExpander"] + div { margin-top: 0 !important; padding-top: 0 !important; }
  /* Kill the default block vertical gap after expanders */
  div[data-testid="stVerticalBlock"] > div:has([data-testid="stExpander"]) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
  }
  div[data-testid="stVerticalBlock"] > div:has(.section-header) {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #16162A;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9999BB;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #FF4D8B !important;
    color: white !important;
  }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: #2A2A3E;
    border: 1px solid #3A3A5A;
    border-radius: 8px;
    color: #E8E8F0;
  }

  /* Section headers */
  .section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #FF4D8B;
    text-transform: uppercase;
    margin: 20px 0 8px 0;
  }

  /* Hit / No-hit badge */
  .hit-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 100px;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .hit { background: linear-gradient(135deg, #1DB954, #0f7a35); color: white; }
  .no-hit { background: linear-gradient(135deg, #FF4D8B, #a0174f); color: white; }

  /* Divider */
  hr { border-color: #2A2A4A; }

  /* Plotly chart bg fix */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #FF4D8B, #c4135f);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  h1, h2, h3 { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
GENRES = ["anime", "chill", "emo", "grunge", "indian", "k-pop", "pop", "pop-film", "sad", "sertanejo"]
GENRE_EMOJI = {
    "anime": "🎌", "chill": "🌊", "emo": "🖤", "grunge": "🎸",
    "indian": "🪘", "k-pop": "⭐", "pop": "🎤", "pop-film": "🎬",
    "sad": "🌧️", "sertanejo": "🤠"
}

# Features Model 1 uses (drops key, mode, time_signature)
FEATURE_COLS = [
    "duration_ms", "explicit", "danceability", "energy",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]
BINARY_COLS  = ["explicit"]
NUMERIC_COLS = [f for f in FEATURE_COLS if f not in BINARY_COLS]

FEATURE_LABELS = {
    "duration_ms":      "Duration (ms)",
    "explicit":         "Explicit Content",
    "danceability":     "Danceability",
    "energy":           "Energy",
    "loudness":         "Loudness (dB)",
    "speechiness":      "Speechiness",
    "acousticness":     "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness":         "Liveness",
    "valence":          "Valence (positivity)",
    "tempo":            "Tempo (BPM)",
}

FEATURE_TIPS = {
    "danceability":     "How suitable the track is for dancing (0 = least, 1 = most)",
    "energy":           "Perceptual intensity and activity (0 = calm, 1 = intense)",
    "loudness":         "Overall loudness in dB (typically -60 to 0)",
    "speechiness":      "Presence of spoken words (> 0.66 = mostly speech)",
    "acousticness":     "Confidence the track is acoustic (0 = electric, 1 = acoustic)",
    "instrumentalness": "Predicts no vocals (> 0.5 = likely instrumental)",
    "liveness":         "Detects live audience presence",
    "valence":          "Musical positivity (0 = sad/angry, 1 = happy/euphoric)",
    "tempo":            "Estimated beats per minute",
    "duration_ms":      "Track length in milliseconds",
    "explicit":         "Whether the track has explicit lyrics",
}

# ── Data & Model loading ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Top_10_genres_Spotify_Music.csv")
    return df

@st.cache_resource
def train_models(df):
    """Train one XGBClassifier per genre, return models + scalers + feature importances."""
    models, scalers, importances = {}, {}, {}

    for genre in GENRES:
        subset = df[df["track_genre"] == genre].copy()
        X = subset[FEATURE_COLS]
        y = subset["is_hit"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_s = X_train.copy()
        X_train_s[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])

        n_pos  = (y_train == 1).sum()
        n_neg  = (y_train == 0).sum()
        spw    = float(n_neg) / n_pos if n_pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X_train_s, y_train)

        fi = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        models[genre]      = model
        scalers[genre]     = scaler
        importances[genre] = fi

    return models, scalers, importances

# ── Plotly theme helper ───────────────────────────────────
PLOT_BG    = "#1E1E2E"
PLOT_PAPER = "#1E1E2E"
GRID_COLOR = "#2A2A4A"
TEXT_COLOR = "#9999BB"
PINK       = "#FF4D8B"
GREEN      = "#1DB954"
TEAL       = "#00B4D8"

def base_layout(**kwargs):
    return dict(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, family="Inter"),
        margin=dict(l=20, r=20, t=40, b=20),
        **kwargs
    )

# ── Load ──────────────────────────────────────────────────
df = load_data()
with st.spinner("Training genre-specific XGBoost models…"):
    models, scalers, importances = train_models(df)

# ── Header ────────────────────────────────────────────────
st.markdown("## Spotify Hit Predictor Dashboard")
st.markdown("<p style='color:#9999BB;margin-top:-10px;'>Choose a genre and adjust audio features to predict whether your song will be a hit.</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────
tab_predict, tab_eda = st.tabs(["🎯  Prediction", "📊  EDA Explorer"])

# ══════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════
with tab_predict:

    # Reserve chart area first (renders at top)
    charts_container = st.container()

    # ── Genre + sliders live here only ────────────────────
    st.markdown("<div style='margin-top:-24px'><div class='section-header'>Genre</div></div>", unsafe_allow_html=True)
    genre = st.selectbox(
        "Select genre",
        GENRES,
        format_func=lambda g: f"{GENRE_EMOJI[g]}  {g.replace('-', ' ').title()}",
        label_visibility="collapsed"
    )

    genre_df   = df[df["track_genre"] == genre]
    genre_mean = genre_df[FEATURE_COLS].mean()

    st.markdown("<div class='section-header' style='margin-top:16px'>Audio Features</div>", unsafe_allow_html=True)

    user_input = {}
    slider_features = [f for f in FEATURE_COLS if f != "explicit"]

    cols = st.columns(3)
    for idx, feat in enumerate(slider_features):
        label = FEATURE_LABELS[feat]
        tip   = FEATURE_TIPS.get(feat, "")
        col   = cols[idx % 3]
        with col:
            if feat == "duration_ms":
                user_input[feat] = st.slider(label, 30000, 600000, int(genre_mean[feat]), step=1000, help=tip)
            elif feat == "loudness":
                user_input[feat] = st.slider(label, -40.0, 0.0, float(round(genre_mean[feat], 1)), step=0.1, help=tip)
            elif feat == "tempo":
                user_input[feat] = st.slider(label, 50.0, 220.0, float(round(genre_mean[feat], 1)), step=0.5, help=tip)
            else:
                user_input[feat] = st.slider(label, 0.0, 1.0, float(round(genre_mean[feat], 3)), step=0.01, help=tip)

    user_input["explicit"] = int(st.checkbox("Explicit Content", value=bool(round(genre_mean["explicit"])), help=FEATURE_TIPS["explicit"]))

    # ── Predict ───────────────────────────────────────────
    input_df = pd.DataFrame([[user_input[f] for f in FEATURE_COLS]], columns=FEATURE_COLS)
    input_scaled = input_df.copy()
    input_scaled[NUMERIC_COLS] = scalers[genre].transform(input_df[NUMERIC_COLS])

    model = models[genre]
    prob  = model.predict_proba(input_scaled)[0][1]
    pred  = int(prob >= 0.5)

    # ── Fill chart container (renders above controls) ─────
    with charts_container:
        st.markdown(f"<p style='color:#9999BB;font-size:13px;margin-bottom:8px'>Powered by genre-specific XGBoost · Model 1 · {GENRE_EMOJI[genre]} {genre.replace('-',' ').title()}</p>", unsafe_allow_html=True)

        # ── Top metrics row ───────────────────────────────────
        genre_hit_rate = genre_df["is_hit"].mean()
        delta = (prob - genre_hit_rate) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Hit Probability", f"{prob*100:.1f}%")
        with c2:
            verdict = "✅ HIT" if pred else "❌ NOT A HIT"
            st.metric("Prediction", verdict)
        with c3:
            st.metric("vs Genre Avg", f"{genre_hit_rate*100:.1f}%", delta=f"{delta:+.1f}%")
        with c4:
            st.metric("Genre Sample Size", f"{len(genre_df):,} tracks")

        st.markdown("---")

        # ── Three charts side by side ─────────────────────────
        col_gauge, col_fi, col_radar = st.columns(3)

        # 1. Gauge
        with col_gauge:
            st.markdown("#### Hit Probability")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                delta={"reference": genre_hit_rate * 100, "suffix": "%", "valueformat": ".1f"},
                number={"suffix": "%", "font": {"size": 32, "color": PINK}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRID_COLOR, "tickfont": {"color": TEXT_COLOR}},
                    "bar": {"color": PINK, "thickness": 0.25},
                    "bgcolor": PLOT_BG,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "#252535"},
                        {"range": [50, 100], "color": "#1a2e1a"},
                    ],
                    "threshold": {
                        "line": {"color": GREEN, "width": 3},
                        "thickness": 0.8,
                        "value": 50
                    }
                },
                title={"text": f"<span style='color:{TEXT_COLOR};font-size:12px'>Genre avg: {genre_hit_rate*100:.1f}%</span>"}
            ))
            fig_gauge.update_layout(**base_layout(height=380))
            st.plotly_chart(fig_gauge, use_container_width=True)
            badge_class = "hit" if pred else "no-hit"
            badge_text  = "🎵 HIT" if pred else "💤 NOT A HIT"
            st.markdown(f"<div style='text-align:center;margin-top:4px;margin-bottom:24px'><span class='hit-badge {badge_class}'>{badge_text}</span></div>", unsafe_allow_html=True)

        # 2. Feature importance
        with col_fi:
            st.markdown("#### Feature Importance")
            fi = importances[genre].copy()
            fi["Label"] = fi["Feature"].map(FEATURE_LABELS)
            fi_sorted = fi.sort_values("Importance")
            fig_fi = go.Figure(go.Bar(
                x=fi_sorted["Importance"],
                y=fi_sorted["Label"],
                orientation="h",
                marker=dict(
                    color=[PINK if fi_sorted["Importance"].iloc[i] == fi_sorted["Importance"].max()
                           else TEAL if fi_sorted["Importance"].iloc[i] >= fi_sorted["Importance"].quantile(0.6)
                           else "#3A3A6A"
                           for i in range(len(fi_sorted))],
                    line=dict(width=0)
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            ))
            fig_fi.update_layout(
                **base_layout(height=420),
                xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title="Importance Gain", color=TEXT_COLOR, tickfont=dict(size=10)),
                yaxis=dict(showgrid=False, color=TEXT_COLOR, tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # 3. Radar
        with col_radar:
            st.markdown("#### Your Track vs Genre Avg")
            radar_features = ["danceability", "energy", "speechiness", "acousticness",
                              "instrumentalness", "liveness", "valence"]
            radar_labels = [FEATURE_LABELS[f] for f in radar_features]
            user_vals = [user_input[f] for f in radar_features]
            avg_vals  = [genre_mean[f] for f in radar_features]
            hit_vals  = [genre_df[genre_df["is_hit"] == 1][f].mean() for f in radar_features]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_vals + [avg_vals[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", fillcolor="rgba(153,153,187,0.15)",
                line=dict(color="#9999BB", width=1.5, dash="dash"),
                name="Genre Avg"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=hit_vals + [hit_vals[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", fillcolor="rgba(29,185,84,0.1)",
                line=dict(color=GREEN, width=1.5, dash="dot"),
                name="Hit Profile"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=user_vals + [user_vals[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", fillcolor="rgba(255,77,139,0.2)",
                line=dict(color=PINK, width=2.5),
                name="Your Track"
            ))
            fig_radar.update_layout(
                **base_layout(height=420),
                polar=dict(
                    bgcolor="#252535",
                    radialaxis=dict(visible=True, range=[0, 1], color=GRID_COLOR, gridcolor=GRID_COLOR, tickfont=dict(size=8)),
                    angularaxis=dict(color=TEXT_COLOR, gridcolor=GRID_COLOR, tickfont=dict(size=9)),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
                            font=dict(color=TEXT_COLOR, size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Feature value table ───────────────────────────────
        with st.expander("📋  View all feature values"):
            display_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
                "Your Value": [user_input[f] for f in FEATURE_COLS],
                "Genre Average": [round(genre_mean[f], 4) for f in FEATURE_COLS],
                "Hit Average": [round(genre_df[genre_df["is_hit"] == 1][f].mean(), 4) for f in FEATURE_COLS],
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("")

    # ── Popularity distribution across genres ─────────────
    st.markdown("#### Average Popularity by Genre")

    genre_stats = df.groupby("track_genre").agg(
        avg_popularity=("popularity", "mean"),
        hit_rate=("is_hit", "mean"),
        count=("is_hit", "count")
    ).reset_index().sort_values("avg_popularity", ascending=False)
    genre_stats["label"] = genre_stats["track_genre"].apply(
        lambda g: f"{GENRE_EMOJI.get(g,'🎵')} {g.replace('-',' ').title()}"
    )

    col_bar, col_hit = st.columns(2)

    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=genre_stats["avg_popularity"],
            y=genre_stats["label"],
            orientation="h",
            marker=dict(
                color=genre_stats["avg_popularity"],
                colorscale=[[0, "#2A2A6E"], [0.5, TEAL], [1, PINK]],
                showscale=False,
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b><br>Avg Popularity: %{x:.1f}<extra></extra>"
        ))
        fig_bar.update_layout(
            **base_layout(height=360, title="Average Popularity Score"),
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, color=TEXT_COLOR, range=[0, 100]),
            yaxis=dict(showgrid=False, color=TEXT_COLOR),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_hit:
        fig_hit = go.Figure(go.Bar(
            x=genre_stats["hit_rate"] * 100,
            y=genre_stats["label"],
            orientation="h",
            marker=dict(
                color=genre_stats["hit_rate"],
                colorscale=[[0, "#2A2A6E"], [0.5, GREEN], [1, PINK]],
                showscale=False,
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b><br>Hit Rate: %{x:.1f}%<extra></extra>"
        ))
        fig_hit.update_layout(
            **base_layout(height=360, title="Hit Rate by Genre (%)"),
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, color=TEXT_COLOR, title="%", range=[0, 100]),
            yaxis=dict(showgrid=False, color=TEXT_COLOR),
        )
        st.plotly_chart(fig_hit, use_container_width=True)



# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4A4A6A;font-size:12px;'>IEOR 142 Final Project · Eden Huang · Vivian Tran · Kevin Jiang · Un Ieng Sit · Ashley Chavez · Gyssell Perez</p>",
    unsafe_allow_html=True
)
