# AI-Based Road Accident Risk Prediction & Hotspot Analysis System

Machine-learning system that (1) predicts how likely an accident is to be severe for given
road, weather and time conditions, and (2) finds accident hotspots on a map.

## Project structure
```
road-accident-risk-prediction/
├── main.py                  # runs the full pipeline
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/                 # put your accidents.csv here (optional)
│   └── processed/           # cleaned data (generated)
├── models/                  # trained model (generated)
├── outputs/                 # metrics, charts, hotspot_map.html (generated)
├── src/
│   ├── config.py            # paths, constants, risk thresholds
│   ├── data_collection.py   # 1. load CSV or generate synthetic data
│   ├── preprocessing.py     # 2. cleaning
│   ├── feature_engineering.py  # 3. features
│   ├── train_model.py       # 4. train and compare models
│   ├── evaluate.py          # 5. metrics and plots
│   ├── predict.py           # 6. risk probability and level
│   ├── hotspot_analysis.py  # 7. DBSCAN + KDE hotspots
│   └── map_visualization.py # 8. folium map
├── app/
│   └── dashboard.py         # 9. Streamlit dashboard
└── tests/
    └── test_pipeline.py
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main.py                   # data -> clean -> train -> evaluate -> hotspots -> map
streamlit run app/dashboard.py   # interactive dashboard
python -m src.predict            # single example prediction
pytest                           # run tests
```
Open `outputs/hotspot_map.html` in a browser to see the hotspot map.

## Using real data
Place a CSV at `data/raw/accidents.csv` with these columns:
`latitude, longitude, hour, day_of_week (0=Mon), month, weather, road_type, lighting,
vehicle_type, traffic_density, speed_limit, is_holiday, severe (0/1)`.
Sources: MoRTH / data.gov.in, Kaggle "US Accidents", UK STATS19. Rename and map your columns first.
Without a CSV, the project generates synthetic data so everything still runs.

## Risk levels
| Probability of severe accident | Level |
|---|---|
| < 25% | Low |
| 25-50% | Moderate |
| 50-75% | High |
| > 75% | Critical |
