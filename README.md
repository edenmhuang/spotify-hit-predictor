# 🎵 Spotify Hit Predictor — Dashboard

Interactive tool for artists to explore genre-specific hit predictions using XGBoost.

## Setup

```bash
# 1. Put the data file in the dashboard folder
cp Top_10_genres_Spotify_Music.csv dashboard/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## Deploy Free on Streamlit Community Cloud

1. Push the `dashboard/` folder to a GitHub repo
2. Go to **https://share.streamlit.io**
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done, live URL in ~2 minutes

> Make sure `Top_10_genres_Spotify_Music.csv` is committed to the repo alongside `app.py`.

---

## Features

### 🎯 Prediction Tab
- **Genre selector** — loads the corresponding trained XGBoost model
- **Audio feature sliders** — all 11 features with genre-average defaults
- **Hit probability gauge** — shows probability vs genre average
- **Feature importance chart** — which features matter most for this genre
- **Radar chart** — your track vs genre average vs average hit profile
- **Feature table** — full comparison of all values

### 📊 EDA Explorer Tab
- Average popularity and hit rate by genre
- Feature vs Popularity scatter plot (filterable by genre)
- Audio feature correlation heatmap (filterable by genre)
- Hit vs Non-Hit distribution comparison

---

## Project
IEOR 142 Final Project · UC Berkeley
