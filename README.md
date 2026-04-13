# 🚌 Public Transport Delay Propagation & Bottleneck Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Data%20Mining-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Classification%20%7C%20Clustering%20%7C%20LSTM-4CAF50?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Visualization-Power%20BI%20%7C%20ggplot2-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" />
</p>

> **Semester 6 — Data Warehousing & Data Mining Project**  
> An end-to-end data mining pipeline to identify transport bottlenecks, predict trip delays, model delay propagation across a transit network, and forecast future delays using deep learning.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Pipeline Phases](#-pipeline-phases)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Output Artifacts](#-output-artifacts)
- [Technologies Used](#-technologies-used)

---

## 🎯 Overview

Urban public transport networks are prone to cascading delays — a single bottleneck at one station can ripple across the entire network. This project applies a full **data mining & machine learning pipeline** to:

1. **Profile & clean** raw trip delay records
2. **Engineer features** that capture weather, traffic, events, and time-of-day context
3. **Discover delay clusters** to identify bottleneck-prone segments of the network
4. **Classify delays** using Logistic Regression, Random Forest, and SVM
5. **Forecast future hourly delays** using an LSTM neural network
6. **Map delay propagation** across 50 stations using graph-theoretic network analysis

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source file** | `public_transport_delays.csv` |
| **Records** | 2,001 trip observations |
| **Date range** | 2023-01-01 onwards (~8+ days of 15-min interval trips) |
| **Transport modes** | Bus, Metro, Train, Tram |
| **Routes** | 20 routes (Route_1 → Route_20) |
| **Stations** | 50 stations (Station_1 → Station_50) |
| **Weather conditions** | Clear, Cloudy, Fog, Rain, Snow, Storm |
| **Events** | None, Concert, Festival, Parade, Protest, Sports |
| **Seasons** | Winter, Spring, Summer, Autumn |
| **Target variable** | `delayed` — binary (0 = On-Time, 1 = Delayed) |

### Dataset Schema (24 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `trip_id` | Identifier | Unique trip ID (T00000, T00001, …) |
| 2 | `date` | Date | Trip date (YYYY-MM-DD) |
| 3 | `time` | Time | Scheduled start time |
| 4 | `transport_type` | Categorical | Bus / Metro / Train / Tram |
| 5 | `route_id` | Categorical | Route identifier |
| 6 | `origin_station` | Categorical | Departure station |
| 7 | `destination_station` | Categorical | Arrival station |
| 8 | `scheduled_departure` | Time | Scheduled departure time |
| 9 | `scheduled_arrival` | Time | Scheduled arrival time |
| 10 | `actual_departure_delay_min` | Numeric | Departure delay in minutes (negative = early) |
| 11 | `actual_arrival_delay_min` | Numeric | Arrival delay in minutes |
| 12 | `weather_condition` | Categorical | Weather at trip time |
| 13 | `temperature_C` | Numeric | Temperature in °C (−5 to 35) |
| 14 | `humidity_percent` | Numeric | Humidity (30–100%) |
| 15 | `wind_speed_kmh` | Numeric | Wind speed (0–60 km/h) |
| 16 | `precipitation_mm` | Numeric | Precipitation (0–20 mm) |
| 17 | `event_type` | Categorical | Nearby event type |
| 18 | `event_attendance_est` | Numeric | Estimated event attendance (0–50,000) |
| 19 | `traffic_congestion_index` | Numeric | Congestion score (0–100) |
| 20 | `holiday` | Binary | 1 = public holiday |
| 21 | `peak_hour` | Binary | 1 = rush hour |
| 22 | `weekday` | Numeric | 0 (Sunday) → 6 (Saturday) |
| 23 | `season` | Categorical | Season of the year |
| 24 | `delayed` | Binary | **Target** — 1 = delayed, 0 = on-time |

---

## 🏗️ Project Architecture

```
public-transport-bottleneck/
│
├── data/
│   └── public_transport_delays.csv       # Raw dataset
│
├── scripts/                              # R analysis pipeline
│   ├── 00_install_packages.R             # Dependency installer
│   ├── 01_load_and_profile.R             # Data loading & profiling
│   ├── 02_data_cleaning.R                # Missing values, outliers, deduplication
│   ├── 03_transformation.R               # Feature engineering & encoding
│   ├── 04_eda.R                          # Exploratory data analysis (40+ plots)
│   ├── 05_pca.R                          # Dimensionality reduction (PCA)
│   ├── 06_clustering.R                   # K-Means & DBSCAN clustering
│   ├── 07_classification.R               # LR, Random Forest, SVM models
│   ├── 08_lstm.R                         # LSTM deep learning forecaster
│   ├── 09_propagation.R                  # Network graph analysis
│   ├── 10_reporting.R                    # Power BI exports & summary reports
│   └── run_all.R                         # 🚀 Master runner — executes all phases
│
├── output/
│   ├── cleaned/                          # Processed datasets (.csv, .rds)
│   ├── plots/                            # 47 generated visualizations (.png)
│   ├── models/                           # Trained model objects (.rds)
│   └── reports/                          # Summary tables & Power BI CSVs
│
└── docs/                                 # Detailed phase-by-phase documentation
    ├── 01_load_and_profile.md
    ├── 02_data_cleaning.md
    └── ...
```

---

## ⚙️ Pipeline Phases

```
Raw CSV ──► [Phase 1] Load & Profile
              │
              ▼
         [Phase 2] Data Cleaning
         (imputation, dedup, outlier capping)
              │
              ▼
         [Phase 3] Feature Engineering
         (temporal, delay, weather & event features)
              │
              ▼
         [Phase 4] EDA ──────────────────── 40+ plots
              │
              ▼
         [Phase 5] PCA (dimensionality reduction)
              │
              ▼
         [Phase 6] Clustering ──────────── Bottleneck discovery
         (K-Means + DBSCAN)
              │
              ▼
         [Phase 7] Classification ──────── Delay prediction
         (Logistic Regression, Random Forest, SVM)
              │
              ▼
         [Phase 8] LSTM ─────────────────  Hourly delay forecasting
              │
              ▼
         [Phase 9] Network Analysis ─────  Delay propagation graph
              │
              ▼
         [Phase 10] Reporting ───────────  Power BI exports
```

### Phase Descriptions

| Phase | Script | Purpose |
|-------|--------|---------|
| **0** | `00_install_packages.R` | Auto-installs all required CRAN packages |
| **1** | `01_load_and_profile.R` | Loads dataset, generates data profile report |
| **2** | `02_data_cleaning.R` | Handles missing values, removes duplicates, caps outliers via IQR winsorization |
| **3** | `03_transformation.R` | Creates 15+ engineered features: temporal buckets, delay composites, weather severity index, event impact score |
| **4** | `04_eda.R` | Produces 40+ plots covering delay distributions, weather effects, station heatmaps, and correlations |
| **5** | `05_pca.R` | Applies PCA (Kaiser criterion + scree elbow), reduces to top components retaining ≥85% variance |
| **6** | `06_clustering.R` | Runs K-Means (elbow + silhouette) and DBSCAN; profiles each cluster by delay severity |
| **7** | `07_classification.R` | Trains 3 ML classifiers with 5-fold CV; compares accuracy, F1, AUC; saves best model |
| **8** | `08_lstm.R` | Builds an hourly time-series from trip data; trains LSTM (50 units, Dropout, Dense) for delay forecasting |
| **9** | `09_propagation.R` | Constructs igraph network of station–station delay propagation; computes PageRank bottleneck scores |
| **10** | `10_reporting.R` | Exports Power BI-ready CSVs and a complete output manifest |

---

## 📊 Key Results

### 🎯 Clustering — Bottleneck Discovery

Three distinct delay clusters were identified from the transit data:

| Cluster | Label | Observations | Avg Arrival Delay | Delay Rate | Dominant Mode |
|---------|-------|:---:|:---:|:---:|:---:|
| **C2** | 🟢 On-Time / Stable | 834 | 5.2 min | 48.7% | Metro |
| **C1** | 🟡 Moderate / Delay-Sensitive | 343 | 14.3 min | 78.7% | Bus |
| **C3** | 🔴 Bottleneck-Prone / High-Delay | 823 | **21.2 min** | **100%** | Bus |

> **Finding:** Cluster 3 (bottleneck-prone) shows 100% delay rate with an average 21-minute arrival delay — driven by high traffic congestion and adverse weather combinations.

---

### 🤖 Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|:--------:|:---------:|:------:|:--------:|:---:|
| Logistic Regression | 47.9% | 71.8% | 50.2% | 59.1% | 0.563 |
| **Random Forest** | **73.2%** | **75.4%** | **95.3%** | **84.2%** | 0.518 |
| SVM (Radial) | 51.9% | 76.9% | 51.2% | 61.5% | 0.525 |

> **Best model:** Random Forest — 73.2% accuracy with **95.3% recall**, making it ideal for operational use where missing a delayed trip is costly.

---

### 🧠 LSTM Forecasting Performance

| Metric | Value |
|--------|-------|
| MSE | 87.35 |
| RMSE | 9.35 min |
| MAE | 7.20 min |
| MAPE | 53.95% |

> The LSTM was trained on an hourly aggregated delay time series (24-step lookback window) to forecast near-future average delays across the network.

---

### 🌐 Network Analysis — Top Delay Propagators

Delay propagation was modeled as a directed graph (50 nodes, weighted edges by delay amplification). **PageRank** was used to identify the most influential bottleneck stations — those whose delays cascade most strongly through the network.

---

## 🚀 Getting Started

### Prerequisites

- **R** ≥ 4.2.0
- **RStudio** (recommended) or any R-compatible IDE
- Internet connection (for package installation on first run)

### Installation & Execution

**1. Clone the repository**
```bash
git clone https://github.com/MayankV004/public-transport-bottleneck.git
cd public-transport-bottleneck
```

**2. Open R / RStudio and set the working directory**
```r
setwd("/path/to/public-transport-bottleneck")
```

**3. Install all dependencies (first run only)**
```r
source("scripts/00_install_packages.R")
```

**4. Run the complete pipeline**
```r
source("scripts/run_all.R")
```

> ⏱️ Full pipeline runtime is approximately **10–20 minutes** depending on hardware. Each phase logs its progress to the console and saves outputs automatically.

**5. Run individual phases (optional)**
```r
source("scripts/01_load_and_profile.R")   # Phase 1 only
source("scripts/04_eda.R")                # EDA only
source("scripts/07_classification.R")    # ML only
```

---

## 📁 Output Artifacts

All outputs are saved under `output/` and auto-created by the pipeline:

### Cleaned Data (`output/cleaned/`)
| File | Description |
|------|-------------|
| `transport_delays_cleaned.csv` | Cleaned dataset after imputation & outlier handling |
| `transport_delays_transformed.csv` | Feature-engineered dataset |
| `pca_transformed_data.csv` | PCA-reduced feature matrix |
| `clustered_data.csv` | Dataset with cluster labels |
| `hourly_delay_timeseries.csv` | Hourly aggregated delay series for LSTM |

### Models (`output/models/`)
| File | Description |
|------|-------------|
| `pca_model.rds` | Fitted PCA object |
| `kmeans_model.rds` | K-Means cluster model |
| `dbscan_model.rds` | DBSCAN model |
| `model_logistic_regression.rds` | Trained LR classifier |
| `model_random_forest.rds` | Trained RF classifier (best model) |
| `model_svm_radial.rds` | Trained SVM classifier |
| `best_classification_model.rds` | Best model (Random Forest) |
| `lstm_model_fallback.rds` | LSTM model weights |

### Reports (`output/reports/`)
| File | Description |
|------|-------------|
| `data_profile.csv` | Column-level data quality report |
| `cleaning_log.csv` | Audit trail of every cleaning step |
| `model_comparison.csv` | Classification model metrics |
| `cluster_profiles.csv` | Cluster summary statistics |
| `station_network_metrics.csv` | PageRank / centrality per station |
| `powerbi_*.csv` | Power BI-ready exports (6 files) |

### Visualizations (`output/plots/`)

47 plots are generated, including:

| Category | Plots |
|----------|-------|
| **Delay Distributions** | Histograms, boxplots by transport type & weekday |
| **Temporal Patterns** | Hourly line plots, hour×weekday heatmap, daily trend |
| **Weather Impact** | Bar charts, scatter plots (precipitation, temp, wind) |
| **Station Analysis** | Top-15 origin/destination bars, station–hour heatmap |
| **PCA** | Scree plot, biplot, variable contributions |
| **Clustering** | Elbow & silhouette curves, PC1×PC2 scatter, DBSCAN plot |
| **Classification** | Confusion matrices, ROC curves, RF variable importance |
| **LSTM** | Actual vs predicted, residual distribution, error scatter |
| **Network** | Delay propagation graph, top PageRank propagators |

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| **R 4.2+** | Primary analysis language |
| `dplyr`, `tidyr`, `lubridate` | Data wrangling & transformation |
| `ggplot2`, `corrplot`, `pheatmap` | Static visualizations |
| `plotly`, `networkD3` | Interactive visualizations |
| `FactoMineR`, `factoextra` | PCA computation & visualization |
| `cluster`, `dbscan`, `fpc` | Clustering algorithms & validation |
| `caret`, `randomForest`, `e1071` | ML training framework |
| `pROC`, `ROCR` | ROC curve & AUC computation |
| `keras` / `tensorflow` | LSTM deep learning model |
| `igraph` | Network graph construction |
| **Power BI** | Dashboard creation |




