# Spotify Hit Predictor

Can audio features alone predict whether a song will be a hit?
That was the question our team set out to answer for our IEOR 142 
final project at UC Berkeley.

Short answer: not really, but it's more nuanced than that.

Audio features like danceability, energy, and tempo can give you 
a sense of whether a song has the right qualities, but they can't 
tell you how popular it'll actually get. What surprised me most 
was how much song data is out there, over 114,000 tracks, and 
even with all of that, audio features alone aren't enough. 

My take is that combining audio features with user behavior data, 
social media signals (TikTok especially), and marketing data would 
give a much stronger picture of what makes a song pop off.

## What we built

An interactive Streamlit dashboard where you can select a genre, 
adjust audio features, and see how likely your song is to be a hit 
compared to the genre average. We trained a separate XGBoost model 
for each of the top 10 Spotify genres since what makes a hit pop 
song is pretty different from what makes a hit metal song.

## Demo

[[Watch the 1-minute demo](https://youtu.be/6m1t_k_lsu4)]

## Team

Built by Eden, Vivian, Gyssell, Ashley, Michelle, and Kevin
IEOR 142 Final Project · UC Berkeley · Spring 2026

## Project Structure

| Folder | Description |
|--------|-------------|
| data/ | Raw and prepared Spotify datasets |
| notebooks/ | EDA, feature selection, and modeling notebooks |
| app.py | Streamlit dashboard |

## Pipeline

1. EDA: explored 114k tracks across 114 genres
2. Feature Selection: narrowed to top 10 genres, ran VIF analysis
3. Modeling: trained XGBoost and Random Forest models per genre
4. Dashboard: built interactive Streamlit prediction tool

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python, Streamlit, XGBoost, Scikit-learn, Pandas, Plotly
