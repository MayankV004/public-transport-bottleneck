# 📋 DETAILED IMPLEMENTATION PLAN
# Public Transport Delay Analysis — Data Mining & Prediction using R

---

## 🎯 PROJECT OVERVIEW

**Title:** Public Transport Delay Propagation and Bottleneck Prediction  
**Course:** Data Warehousing & Data Mining — Semester 6 Project  
**Language:** R (primary), Power BI (dashboards)  
**Dataset:** `public_transport_delays.csv` (single file, 2001 records)

### Dataset Quick Profile

| Property               | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| Records                | 2001 trip observations                                                  |
| Date Range             | 2023-01-01 onwards (~8+ days of 15-min interval trips)                 |
| Transport Types        | Bus, Metro, Train, Tram (4 categories)                                 |
| Routes                 | Route_1 to Route_20 (20 routes)                                        |
| Stations               | Station_1 to Station_50 (50 stations)                                  |
| Weather Conditions     | Clear, Cloudy, Fog, Rain, Snow, Storm (6 categories)                   |
| Events                 | None, Concert, Festival, Parade, Protest, Sports (6 categories)        |
| Seasons                | Winter, Spring, Summer, Autumn (4 categories)                          |
| Target Variable        | `delayed` (binary: 0 = On-Time, 1 = Delayed) — **pre-labelled**       |

### Columns in the Dataset (24 columns)

| #  | Column                      | Type        | Description                                                 |
|----|-----------------------------|-------------|-------------------------------------------------------------|
| 1  | `trip_id`                   | Identifier  | Unique trip identifier (T00000, T00001, …)                  |
| 2  | `date`                      | Date        | Date of the trip (YYYY-MM-DD)                               |
| 3  | `time`                      | Time        | Scheduled start time (HH:MM:SS)                             |
| 4  | `transport_type`            | Categorical | Bus / Metro / Train / Tram                                  |
| 5  | `route_id`                  | Categorical | Route identifier (Route_1 … Route_20)                      |
| 6  | `origin_station`            | Categorical | Departure station name                                      |
| 7  | `destination_station`       | Categorical | Arrival station name                                        |
| 8  | `scheduled_departure`       | Time        | Scheduled departure time (HH:MM:SS)                         |
| 9  | `scheduled_arrival`         | Time        | Scheduled arrival time (HH:MM:SS)                           |
| 10 | `actual_departure_delay_min`| Numeric     | Delay at departure in minutes (can be negative = early)     |
| 11 | `actual_arrival_delay_min`  | Numeric     | Delay at arrival in minutes (can be negative = early)       |
| 12 | `weather_condition`         | Categorical | Clear / Cloudy / Fog / Rain / Snow / Storm                  |
| 13 | `temperature_C`             | Numeric     | Temperature in Celsius (-5 to 35)                           |
| 14 | `humidity_percent`          | Numeric     | Humidity percentage (30 – 100)                              |
| 15 | `wind_speed_kmh`            | Numeric     | Wind speed in km/h (0 – 60)                                 |
| 16 | `precipitation_mm`          | Numeric     | Precipitation in mm (0 – 20)                                |
| 17 | `event_type`                | Categorical | Type of nearby event (None / Concert / Festival / …)        |
| 18 | `event_attendance_est`      | Numeric     | Estimated attendance (0 – 50000)                            |
| 19 | `traffic_congestion_index`  | Numeric     | Congestion index (0 – 100)                                  |
| 20 | `holiday`                   | Binary      | 1 = holiday, 0 = not                                        |
| 21 | `peak_hour`                 | Binary      | 1 = peak hour, 0 = off-peak                                |
| 22 | `weekday`                   | Numeric     | Day of week as integer (0 = Sunday … 6 = Saturday)          |
| 23 | `season`                    | Categorical | Winter / Spring / Summer / Autumn                           |
| 24 | `delayed`                   | Binary      | **Target** — 1 = delayed, 0 = on-time                      |

### Key Deliverables

| #  | Deliverable                                   | Format              |
|----|-----------------------------------------------|---------------------|
| 1  | Cleaned & transformed dataset                 | CSV                 |
| 2  | Exploratory Data Analysis (EDA) report        | Plots + tables      |
| 3  | Dimensionality Reduction (PCA) results        | Plots + CSV         |
| 4  | Clustering analysis for bottleneck discovery   | Plots + CSV         |
| 5  | ML classification models (RF, SVM, LR)        | .rds model files    |
| 6  | Deep Learning (LSTM) forecasting model         | .h5 model file      |
| 7  | Model comparison report                       | Tables + ROC plots  |
| 8  | Power BI dashboards                           | .pbix               |
| 9  | Final project report                          | .md / .pdf          |

---

## 📚 PHASE 0: ENVIRONMENT SETUP & PACKAGE INSTALLATION

### Step 0.1: Install Required R Packages

**Packages needed and their purpose:**

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `dplyr`        | Data wrangling and transformation                |
| `tidyr`        | Reshaping data (pivot, fill, etc.)               |
| `lubridate`    | Date/time parsing and extraction                 |
| `readr`        | Fast CSV reading/writing                         |
| `ggplot2`      | All static visualisations                        |
| `corrplot`     | Correlation matrix heatmaps                      |
| `plotly`       | Interactive 3-D scatter plots                    |
| `pheatmap`     | Station–hour delay heatmaps                      |
| `factoextra`   | PCA scree/biplot visualisation                   |
| `FactoMineR`   | PCA computation                                  |
| `cluster`      | Silhouette analysis                              |
| `dbscan`       | DBSCAN density-based clustering                  |
| `fpc`          | Cluster validation statistics                    |
| `caret`        | Unified ML training/evaluation framework         |
| `randomForest` | Random Forest classifier                         |
| `e1071`        | SVM classifier                                   |
| `MLmetrics`    | Extra ML evaluation metrics                      |
| `keras`        | LSTM deep-learning model                         |
| `tensorflow`   | Backend for keras                                |
| `igraph`       | Network/graph construction for delay propagation |
| `networkD3`    | Interactive network visualisation                |
| `pROC`         | ROC curve and AUC computation                    |
| `ROCR`         | ROC/PR curve utilities                           |
| `gridExtra`    | Combining multiple ggplots into dashboards       |
| `scales`       | Axis formatting helpers                          |

### Step 0.2: Create Project Directory Structure

Create the following folder layout before running any code:

```
Project/
├── data/                       # Raw data
│   └── public_transport_delays.csv
├── output/                     # All generated outputs
│   ├── cleaned/                # Cleaned CSVs
│   ├── plots/                  # PNG/PDF plots
│   ├── models/                 # Saved .rds / .h5 models
│   └── reports/                # Summary tables & CSVs
├── scripts/                    # R script files
│   ├── 01_load_and_profile.R
│   ├── 02_data_cleaning.R
│   ├── 03_transformation.R
│   ├── 04_eda.R
│   ├── 05_pca.R
│   ├── 06_clustering.R
│   ├── 07_classification.R
│   ├── 08_lstm.R
│   ├── 09_propagation.R
│   └── 10_reporting.R
└── Implementation_Plan.md      # This document
```

---

## 📊 PHASE 1: DATA LOADING & PROFILING
**Script:** `01_load_and_profile.R`

### Step 1.1: Load the Dataset

- Read `public_transport_delays.csv` using `readr::read_csv()`.
- This is a **single flat file** — no joins to other tables are needed.
- Set `trip_id` and `route_id` to character type on import to avoid numeric coercion.

### Step 1.2: Inspect Structure

- Run `str()` and `glimpse()` to verify column types.
- Confirm 2001 rows × 24 columns.
- Use `summary()` on all columns to get quick min/max/mean/median for numerics, and frequency counts for categoricals.

### Step 1.3: Data Profiling Report

Generate a profiling summary containing:

| Check                          | What to Compute                                               |
|--------------------------------|---------------------------------------------------------------|
| Row count                      | `nrow(df)`                                                    |
| Column count                   | `ncol(df)`                                                    |
| Missing values per column      | `colSums(is.na(df))` and percentage                          |
| Unique values per column       | `sapply(df, n_distinct)`                                      |
| Numeric column ranges          | min, max, mean, median, sd for each numeric column            |
| Categorical frequency tables   | `table()` for `transport_type`, `weather_condition`, `event_type`, `season` |
| Date range                     | min/max of `date`                                             |
| Class balance of target        | `table(df$delayed)` → proportion of 0 vs 1                   |

Save this profiling table as `output/reports/data_profile.csv`.

### Step 1.4: Initial Observations to Document

- Are there any columns that are entirely NA?
- What is the class distribution of `delayed`? (is it balanced or imbalanced?)
- Do negative values in `actual_departure_delay_min` and `actual_arrival_delay_min` make sense? (Yes — they mean early departure/arrival.)
- Are there any `origin_station == destination_station` rows? (Potential data-entry errors.)
- Does `weekday` encoding make sense? (0–6 mapping.)

---

## 🧹 PHASE 2: DATA CLEANING (CRITICAL PHASE)
**Script:** `02_data_cleaning.R`

> **This is the most important phase for a Data Mining project.** Every cleaning decision must be documented with a before/after record count.

### Step 2.1: Handle Missing Values

**Strategy per column type:**

| Column Type          | Strategy                                                      |
|----------------------|---------------------------------------------------------------|
| Identifiers (trip_id)| Drop row if missing (critical key)                           |
| Numeric delays       | Drop row if both departure & arrival delay are NA             |
| Weather numerics     | Impute with column median (temperature, humidity, wind, precip) |
| Categorical columns  | Impute with mode, or create "Unknown" category               |
| Binary flags         | Impute with mode (0 or 1 based on majority)                  |

**Steps:**
1. Count missing values per column → save as `missing_value_summary.csv`.
2. For each strategy applied, log: column name, method used, number of values imputed/dropped.
3. After all imputation, verify `colSums(is.na(df_clean)) == 0`.

### Step 2.2: Remove Duplicate Records

- Check for full-row duplicates using `duplicated()`.
- Check for logical duplicates: same `trip_id` appearing more than once.
  - If found, keep the first occurrence.
- Log: number of duplicates found and removed.

### Step 2.3: Validate & Correct Data Types

| Column                       | Expected Type  | Conversion Needed                                  |
|------------------------------|----------------|----------------------------------------------------|
| `date`                       | Date           | `as.Date(date)` — verify format is `YYYY-MM-DD`   |
| `time`                       | Character/Time | Parse with `hms::as_hms()` or keep as character    |
| `scheduled_departure`        | Character/Time | Parse to time or combine with `date` → POSIXct     |
| `scheduled_arrival`          | Character/Time | Parse to time or combine with `date` → POSIXct     |
| `transport_type`             | Factor         | `as.factor()`                                      |
| `weather_condition`          | Factor         | `as.factor()`                                      |
| `event_type`                 | Factor         | `as.factor()`                                      |
| `season`                     | Factor         | `as.factor()`                                      |
| `route_id`                   | Factor         | `as.factor()`                                      |
| `origin_station`             | Factor         | `as.factor()`                                      |
| `destination_station`        | Factor         | `as.factor()`                                      |
| `holiday`, `peak_hour`       | Factor (0/1)   | `as.factor()` for classification later             |
| `delayed`                    | Factor (0/1)   | `as.factor()` — this is the **target**             |
| All numeric columns          | numeric/double | Already correct; verify no character contamination |

### Step 2.4: Detect and Handle Outliers

**Columns to check for outliers:**

| Column                        | Expected Range    | Outlier Rule                                 |
|-------------------------------|-------------------|----------------------------------------------|
| `actual_departure_delay_min`  | -10 to ~20        | Flag values < -10 or > 30 (use IQR method)  |
| `actual_arrival_delay_min`    | -5 to ~30         | Flag values < -10 or > 40 (use IQR method)  |
| `temperature_C`               | -5 to 35          | Flag values outside realistic bounds         |
| `humidity_percent`            | 0 to 100          | Flag values outside 0–100                    |
| `wind_speed_kmh`              | 0 to 60           | Flag extreme wind > 60 km/h                  |
| `precipitation_mm`            | 0 to 20           | Flag negative values                         |
| `traffic_congestion_index`    | 0 to 100          | Flag values outside 0–100                    |
| `event_attendance_est`        | 0 to 50000        | Flag negative values                         |

**Approach:**
1. Use box plots per numeric column to visualize outliers.
2. Apply the IQR rule: outlier if value < Q1 − 1.5×IQR or > Q3 + 1.5×IQR.
3. **Decision:** Cap (winsorize) outliers rather than remove rows to preserve data.
4. Log: number of outliers detected and capped per column.

### Step 2.5: Validate Logical Consistency

Perform these sanity checks:

| Check                                               | Action if Failed                          |
|------------------------------------------------------|-------------------------------------------|
| `origin_station == destination_station`              | Flag as suspicious (circular trip)         |
| `actual_departure_delay_min` very high but `delayed == 0` | Inspect threshold used for labelling    |
| `scheduled_arrival` time earlier than `scheduled_departure` | Could be overnight trip — handle carefully |
| `peak_hour == 1` but `time` is outside 7–9 AM / 4–7 PM | Flag for review                           |
| `holiday == 1` — verify against actual 2023 calendar | Optional cross-reference                  |

### Step 2.6: Save Cleaned Data

- Save the fully cleaned dataframe as `output/cleaned/transport_delays_cleaned.csv`.
- Create and save a **cleaning log** (`output/reports/cleaning_log.csv`) with:

| Step                      | Records Before | Records After | Records Changed | Method               |
|---------------------------|----------------|---------------|-----------------|----------------------|
| Raw data loaded           | 2001           | 2001          | 0               | —                    |
| Missing values handled    | 2001           | …             | …               | Impute/Drop          |
| Duplicates removed        | …              | …             | …               | Keep first           |
| Outliers capped           | …              | …             | …               | IQR winsorization    |
| Invalid records removed   | …              | …             | …               | Logical checks       |
| **Final clean dataset**   | —              | **N**         | —               | —                    |

---

## 🔧 PHASE 3: DATA TRANSFORMATION & FEATURE ENGINEERING (CRITICAL PHASE)
**Script:** `03_transformation.R`

> **Feature engineering drives model performance.** This phase creates meaningful derived features from the raw columns.

### Step 3.1: Create Datetime Columns

- Combine `date` + `time` → `datetime` (POSIXct) for full timestamp.
- Combine `date` + `scheduled_departure` → `sched_dep_datetime`.
- Combine `date` + `scheduled_arrival` → `sched_arr_datetime`.
- **Handle overnight trips:** If `scheduled_arrival` < `scheduled_departure`, add +1 day to arrival datetime.

### Step 3.2: Extract Temporal Features

From the `datetime` column, extract:

| New Feature          | Derivation                                           | Purpose                                |
|----------------------|------------------------------------------------------|----------------------------------------|
| `hour`               | `lubridate::hour(datetime)`                          | Hourly delay patterns                  |
| `day_of_week`        | Already available as `weekday` (0–6)                 | Daily patterns                         |
| `day_name`           | `weekdays(date)` → Mon, Tue, …                      | Readable day labels                    |
| `is_weekend`         | `weekday %in% c(0, 6)` → 1/0                        | Weekday vs weekend effect              |
| `month`              | `lubridate::month(date)`                             | Monthly trends                         |
| `week_of_year`       | `lubridate::week(date)`                              | Weekly seasonality                     |
| `time_of_day_bucket` | Group `hour` into: Night(0–5), Morning(6–11), Afternoon(12–17), Evening(18–23) | Coarser time grouping |

### Step 3.3: Compute Trip-Level Delay Features

| New Feature              | Formula                                                              | Purpose                              |
|--------------------------|----------------------------------------------------------------------|--------------------------------------|
| `total_delay`            | `actual_departure_delay_min + actual_arrival_delay_min`              | Combined trip delay score            |
| `delay_change`           | `actual_arrival_delay_min - actual_departure_delay_min`              | Did delay grow or shrink en-route?   |
| `is_early_departure`     | `actual_departure_delay_min < 0` → 1/0                              | Left before schedule                 |
| `is_early_arrival`       | `actual_arrival_delay_min < 0` → 1/0                                | Arrived before schedule              |
| `trip_duration_scheduled` | `sched_arr_datetime - sched_dep_datetime` (minutes)                 | Expected travel time                 |
| `severe_delay`           | `actual_arrival_delay_min > 15` → 1/0                               | Severely delayed trips               |

### Step 3.4: Encode Categorical Variables

**For EDA and clustering (label encoding):**

| Column              | Encoding                                                             |
|----------------------|----------------------------------------------------------------------|
| `transport_type`     | Bus=1, Metro=2, Train=3, Tram=4                                     |
| `weather_condition`  | Clear=1, Cloudy=2, Fog=3, Rain=4, Snow=5, Storm=6 (ordinal by severity) |
| `event_type`         | None=0, Concert=1, Festival=2, Parade=3, Protest=4, Sports=5        |
| `season`             | Winter=1, Spring=2, Summer=3, Autumn=4                               |

**For ML models (one-hot encoding via `caret::dummyVars()`):**
- Create dummy variables for all multi-class categorical columns.
- Avoid dummy trap (drop one level per variable).

### Step 3.5: Create Interaction & Derived Weather Features

| New Feature                | Formula                                           | Rationale                                     |
|----------------------------|---------------------------------------------------|-----------------------------------------------|
| `weather_severity_index`   | weighted combo of wind + precipitation + (storm indicator) | Single weather impact score          |
| `feels_like_impact`        | interaction of `temperature_C` × `wind_speed_kmh` | Wind chill effect on operations               |
| `event_impact_score`       | `event_attendance_est` × `traffic_congestion_index` / 1000 | Combined event disruption measure    |
| `congestion_weather_combo` | `traffic_congestion_index` × `weather_severity_index` | Compounded disruption               |

### Step 3.6: Aggregate by Station (Origin)

Group by `origin_station` and compute:

| Aggregated Feature            | Computation                                       |
|-------------------------------|---------------------------------------------------|
| `station_avg_dep_delay`       | mean of `actual_departure_delay_min`              |
| `station_avg_arr_delay`       | mean of `actual_arrival_delay_min`                |
| `station_delay_rate`          | proportion where `delayed == 1` × 100            |
| `station_total_trips`         | count of trips originating                        |
| `station_sd_delay`            | standard deviation of `actual_arrival_delay_min`  |
| `station_max_delay`           | max of `actual_arrival_delay_min`                 |

Save as `output/reports/station_origin_statistics.csv`.

### Step 3.7: Aggregate by Route

Group by `route_id` and compute:

| Aggregated Feature          | Computation                                         |
|-----------------------------|-----------------------------------------------------|
| `route_avg_delay`           | mean of `actual_arrival_delay_min`                  |
| `route_delay_rate`          | proportion where `delayed == 1` × 100              |
| `route_total_trips`         | count of trips                                      |
| `route_delay_variability`   | sd of `actual_arrival_delay_min`                    |

Save as `output/reports/route_statistics.csv`.

### Step 3.8: Aggregate by Hour × Day Type

Group by `hour` + `is_weekend` and compute:

| Feature                   | Computation                             |
|---------------------------|-----------------------------------------|
| `avg_delay_by_hour_day`   | mean of `actual_arrival_delay_min`      |
| `delay_rate_by_hour_day`  | proportion `delayed == 1`               |
| `trip_count_by_hour_day`  | count                                   |

This will be used for EDA hourly-pattern plots and for the LSTM time-series later.

### Step 3.9: Normalize / Standardize Numeric Features

- **Z-score standardization** (mean=0, sd=1): Use for PCA and clustering.
  - Apply to: `actual_departure_delay_min`, `actual_arrival_delay_min`, `temperature_C`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`, `event_attendance_est`, `traffic_congestion_index`, and all derived numeric features.
- **Min-Max scaling** [0,1]: Use for LSTM deep learning model.
- Save the scaling parameters (mean, sd, min, max) for each column so predictions can be inverse-transformed later.

### Step 3.10: Save Transformed Dataset

- Save the final feature-engineered dataframe as `output/cleaned/transport_delays_transformed.csv`.
- Save a **feature dictionary** (`output/reports/feature_dictionary.csv`) listing every column with its name, type, source, and description.

---

## 📊 PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
**Script:** `04_eda.R`

### Step 4.1: Delay Distribution Analysis

| Plot                                        | Type           | X-axis                        | Y-axis     | Notes                              |
|---------------------------------------------|----------------|-------------------------------|------------|------------------------------------|
| Distribution of `actual_arrival_delay_min`  | Histogram      | Delay (min)                   | Frequency  | Use bins=50, add mean/median lines |
| Distribution of `actual_departure_delay_min`| Histogram      | Delay (min)                   | Frequency  | Compare with arrival delay         |
| Arrival delay by `transport_type`           | Box plot        | Transport type                | Delay (min)| Side-by-side, coloured by type     |
| Arrival delay by `is_weekend`               | Box plot        | Weekday / Weekend             | Delay (min)| Compare patterns                   |

### Step 4.2: Temporal Pattern Analysis

| Plot                                             | Type       | Details                                    |
|--------------------------------------------------|------------|--------------------------------------------|
| Average delay by hour of day                     | Line plot  | X = hour (0–23), Y = avg delay, facet by weekday/weekend |
| Delay rate by day of week                        | Bar chart  | X = day name, Y = % delayed                |
| Heatmap: hour × weekday delay intensity          | Heatmap    | 7 rows (days) × 24 cols (hours), fill = avg arrival delay |
| Delay trend across dates                         | Line plot  | X = date, Y = daily average delay           |

### Step 4.3: Weather Impact Analysis

| Plot                                                | Type           | Details                                       |
|-----------------------------------------------------|----------------|-----------------------------------------------|
| Average delay by `weather_condition`                | Bar chart      | 6 bars, sorted by avg delay                   |
| Delay rate by `weather_condition`                   | Stacked bar    | % delayed vs on-time per weather type          |
| Scatter: `temperature_C` vs `actual_arrival_delay`  | Scatter + trend| Look for non-linear relationships              |
| Scatter: `precipitation_mm` vs `actual_arrival_delay`| Scatter + trend| Expected: higher precip → higher delay         |
| Scatter: `wind_speed_kmh` vs `actual_arrival_delay` | Scatter + trend| Wind impact                                    |

### Step 4.4: Event & Traffic Impact Analysis

| Plot                                                     | Type           | Details                                   |
|----------------------------------------------------------|----------------|-------------------------------------------|
| Average delay by `event_type`                            | Bar chart      | 6 bars, sorted by avg delay               |
| Scatter: `traffic_congestion_index` vs `arrival_delay`   | Scatter + trend| Key predictor?                            |
| Scatter: `event_attendance_est` vs `arrival_delay`       | Scatter        | High-attendance events → more delay?       |
| Box plot: delay when `holiday==1` vs `holiday==0`        | Box plot       | Holiday effect                             |
| Box plot: delay when `peak_hour==1` vs `peak_hour==0`    | Box plot       | Peak-hour effect                           |

### Step 4.5: Station & Route Analysis

| Plot                                            | Type        | Details                                       |
|-------------------------------------------------|-------------|-----------------------------------------------|
| Top 15 origin stations by avg delay             | Horizontal bar | Ranked, coloured by delay severity          |
| Top 15 destination stations by avg delay        | Horizontal bar | Ranked                                      |
| Top 10 routes by avg delay                      | Horizontal bar | Ranked                                      |
| Station–hour heatmap (origin stations)          | Heatmap     | Rows = top 20 stations, Cols = hours, Fill = avg delay |

### Step 4.6: Correlation Analysis

- Select all **numeric** columns (delays, weather, traffic, attendance, derived features).
- Compute Pearson correlation matrix.
- Visualize with `corrplot()` using color method with coefficients displayed.
- **Key correlations to report:**
  - `traffic_congestion_index` ↔ `actual_arrival_delay_min`
  - `precipitation_mm` ↔ `actual_arrival_delay_min`
  - `actual_departure_delay_min` ↔ `actual_arrival_delay_min`
  - `event_attendance_est` ↔ delay

### Step 4.7: Target Variable Analysis

- Bar chart: class distribution of `delayed` (count and %).
- Cross-tabulation: `delayed` vs `transport_type`, `weather_condition`, `season`, `peak_hour`.
- Check for class imbalance — if one class < 30%, plan mitigation in Phase 7.

### Step 4.8: Save All EDA Plots

Save every plot as PNG (300 dpi) into `output/plots/` with descriptive filenames:
- `delay_distribution_histogram.png`
- `delay_by_transport_type.png`
- `hourly_delay_pattern.png`
- `weather_impact_bar.png`
- `station_hour_heatmap.png`
- `correlation_matrix.png`
- `target_class_distribution.png`
- etc.

---

## 📉 PHASE 5: DIMENSIONALITY REDUCTION (PCA)
**Script:** `05_pca.R`

### Step 5.1: Select Features for PCA

Select only **numeric** columns relevant for pattern discovery:

| Feature Group         | Columns                                                                                  |
|-----------------------|------------------------------------------------------------------------------------------|
| Delay metrics         | `actual_departure_delay_min`, `actual_arrival_delay_min`, `total_delay`, `delay_change`  |
| Weather metrics       | `temperature_C`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`                |
| External factors      | `event_attendance_est`, `traffic_congestion_index`                                       |
| Derived features      | `weather_severity_index`, `event_impact_score`, `trip_duration_scheduled`                |
| Pre-computed flags    | `holiday`, `peak_hour`, `weekday`                                                        |

Remove: `trip_id`, `date`, `time`, all character/factor columns, and `delayed` (target).

### Step 5.2: Standardize Features (Z-score)

- Use `scale()` on the selected numeric feature matrix.
- Verify: all column means ≈ 0, all column sds ≈ 1.
- Remove any columns with zero variance (constant columns).

### Step 5.3: Perform PCA

- Use `prcomp()` with `center = FALSE, scale. = FALSE` (already scaled).
- Extract: eigenvalues, proportion of variance explained, cumulative variance.
- Create variance explained table:

| Component | Eigenvalue | Variance Explained (%) | Cumulative (%) |
|-----------|-----------|------------------------|-----------------|
| PC1       | …         | …                      | …               |
| PC2       | …         | …                      | …               |
| …         | …         | …                      | …               |

### Step 5.4: Determine Number of Components

**Three criteria to apply:**
1. **Kaiser criterion:** Keep components with eigenvalue > 1.
2. **Scree plot elbow:** Visual bend in the scree plot.
3. **Cumulative variance ≥ 85–90%:** Retain enough PCs to explain ≥ 85% variance.

Document the chosen number of components and justification.

### Step 5.5: Visualize PCA

| Plot                              | Tool / Function               | Purpose                                       |
|-----------------------------------|-------------------------------|-----------------------------------------------|
| Scree plot                        | `factoextra::fviz_eig()`      | Show variance explained per PC                |
| Biplot (PC1 vs PC2)              | `factoextra::fviz_pca_biplot()` | Show observations + variable loadings        |
| Variable contribution             | `factoextra::fviz_pca_var()`  | Which original features contribute most?      |
| Loadings table                    | `pca_model$rotation`          | Numeric loadings of each variable on each PC  |

### Step 5.6: Save PCA Outputs

- PCA-transformed data (selected components + original identifiers) → `output/cleaned/pca_transformed_data.csv`
- PCA model object → `output/models/pca_model.rds`
- Variance explained table → `output/reports/pca_variance_explained.csv`
- Loadings table → `output/reports/pca_loadings.csv`

**Report statement:** *"PCA reduced N original features to K principal components, retaining X% of the total variance."*

---

## 🎯 PHASE 6: CLUSTERING (BOTTLENECK IDENTIFICATION)
**Script:** `06_clustering.R`

### Step 6.1: Prepare Clustering Input

- Use the PCA-transformed data from Phase 5 (top K components).
- Alternatively, also run clustering on the standardized original features for comparison.
- Remove any remaining NA values.

### Step 6.2: Determine Optimal K (Elbow + Silhouette)

**Elbow Method:**
- Run K-Means for K = 2, 3, 4, …, 10.
- Record total within-cluster sum of squares (WSS) for each K.
- Plot K vs WSS → look for "elbow" bend.

**Silhouette Analysis:**
- For each K (2–10), compute average silhouette width.
- Plot K vs silhouette score → pick K with highest score.

**Gap Statistic (optional):**
- Use `cluster::clusGap()` for a more formal test.

Document the chosen K and reasoning.

### Step 6.3: Run K-Means Clustering

- Use `kmeans()` with `nstart = 25` for stability.
- Assign cluster labels to each observation.
- Compute cluster centers (centroids).

### Step 6.4: Cluster Profiling

For each cluster, compute:

| Metric                        | How                                                              |
|-------------------------------|------------------------------------------------------------------|
| Number of observations        | `count per cluster`                                              |
| Average arrival delay         | `mean(actual_arrival_delay_min)` per cluster                     |
| Average departure delay       | `mean(actual_departure_delay_min)` per cluster                   |
| Delay rate                    | `mean(delayed == 1)` per cluster                                 |
| Dominant transport type       | mode of `transport_type` per cluster                             |
| Average congestion index      | `mean(traffic_congestion_index)` per cluster                     |
| Dominant weather condition    | mode of `weather_condition` per cluster                          |

**Assign semantic labels based on profiles:**
- **Cluster A — "On-Time / Stable"**: Low avg delay, low delay rate.
- **Cluster B — "Moderate / Delay-Sensitive"**: Medium avg delay, moderate delay rate.
- **Cluster C — "Bottleneck-Prone / High-Delay"**: High avg delay, high delay rate, correlated with bad weather / events.

### Step 6.5: Visualize Clusters

| Plot                                  | Type              | Details                                   |
|---------------------------------------|-------------------|-------------------------------------------|
| PC1 vs PC2 coloured by cluster        | Scatter + ellipse | `ggplot` with `stat_ellipse()`            |
| 3D plot (PC1, PC2, PC3) by cluster    | `plotly` 3D       | Interactive rotation                      |
| Cluster size bar chart                | Bar chart         | Count per cluster                         |
| Cluster profile radar/bar chart       | Grouped bar       | Avg delay, congestion, etc. per cluster   |

### Step 6.6: DBSCAN (Alternative Clustering)

- Use `dbscan::dbscan()` on the PCA-transformed data.
- Determine `eps` from the k-distance plot (`dbscan::kNNdistplot()`).
- Identify noise/outlier points (cluster label = 0).
- Compare DBSCAN results with K-Means results.

### Step 6.7: Save Clustering Outputs

- Clustered data with labels → `output/cleaned/clustered_data.csv`
- Cluster profiles → `output/reports/cluster_profiles.csv`
- Elbow + silhouette plots → `output/plots/`
- Cluster scatter plot → `output/plots/`

---

## 🤖 PHASE 7: MACHINE LEARNING — CLASSIFICATION
**Script:** `07_classification.R`

### Step 7.1: Define Target Variable

- **Primary target:** `delayed` column (already binary 0/1).
- Convert to factor with levels `c("No", "Yes")` for `caret` compatibility.
- **Alternative target (from clustering):** Create `bottleneck` = 1 if cluster is "Bottleneck-Prone", else 0. Can train a second set of models on this.

### Step 7.2: Select Features for Classification

Use these feature groups as input (X):

| Feature Group       | Columns                                                                                    |
|---------------------|--------------------------------------------------------------------------------------------|
| Weather             | `temperature_C`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`, `weather_condition` (encoded) |
| External events     | `event_type` (encoded), `event_attendance_est`, `traffic_congestion_index`                |
| Temporal            | `hour`, `weekday`, `is_weekend`, `peak_hour`, `holiday`, `season` (encoded)               |
| Transport           | `transport_type` (encoded)                                                                 |
| Derived features    | `weather_severity_index`, `event_impact_score`, `trip_duration_scheduled`                 |

**Exclude:** `trip_id`, `date`, `time`, `scheduled_departure`, `scheduled_arrival`, `origin_station`, `destination_station`, `actual_departure_delay_min`, `actual_arrival_delay_min` (these are post-hoc — you wouldn't know actual delays before the trip happens).

> **Important design decision:** If the goal is to **predict** whether a trip will be delayed *before it happens*, do NOT include actual delay columns as features. Only include pre-trip-known features (schedule, weather, events, traffic forecast, temporal info).

### Step 7.3: Train-Test Split

- 80% training / 20% testing using `caret::createDataPartition()` with stratification on `delayed`.
- Log sizes: training set N rows, test set M rows.
- Verify class proportions are preserved in both sets.

### Step 7.4: Handle Class Imbalance (if needed)

- If `delayed` is imbalanced (e.g., 70/30 split):
  - Option A: **Up-sampling** minority class in `trainControl(sampling = "up")`.
  - Option B: **Down-sampling** majority class.
  - Option C: **SMOTE** synthetic oversampling.
- Document which method was used and why.

### Step 7.5: Train Model 1 — Logistic Regression (Baseline)

- Method: `method = "glm"`, `family = "binomial"`.
- Use 5-fold cross-validation.
- Metric: ROC (AUC).
- Generate predictions on test set.
- Compute confusion matrix.

### Step 7.6: Train Model 2 — Random Forest

- Method: `method = "rf"`, `ntree = 100`.
- Use 5-fold cross-validation.
- Metric: ROC (AUC).
- Extract and plot **variable importance** → which features predict delay the best?
- Generate predictions and confusion matrix.

### Step 7.7: Train Model 3 — Support Vector Machine (SVM)

- Method: `method = "svmRadial"`, `tuneLength = 5`.
- Use 5-fold cross-validation.
- Metric: ROC (AUC).
- Generate predictions and confusion matrix.

### Step 7.8: Model Comparison

Create a comparison table:

| Model                | Accuracy | Precision | Recall | F1 Score | Specificity | AUC   |
|----------------------|----------|-----------|--------|----------|-------------|-------|
| Logistic Regression  | …        | …         | …      | …        | …           | …     |
| Random Forest        | …        | …         | …      | …        | …           | …     |
| SVM                  | …        | …         | …      | …        | …           | …     |

**Visualizations:**
- ROC curves for all 3 models overlaid on one plot with AUC in legend.
- Confusion matrix heatmaps for each model.
- Variable importance bar chart (from Random Forest).
- Model comparison grouped bar chart (Accuracy, F1, AUC side by side).

### Step 7.9: Select & Save Best Model

- Pick best model based on **F1-score** (balance between precision and recall) or **AUC**.
- Justify: "High recall is important to avoid missing delayed trips (false negatives are costly for operations)."
- Save: `output/models/best_classification_model.rds`
- Save: `output/reports/model_comparison.csv`

---

## 🧠 PHASE 8: DEEP LEARNING — LSTM DELAY FORECASTING
**Script:** `08_lstm.R`

### Step 8.1: Create Time-Series Dataset

- Sort the data by `date` + `time`.
- Aggregate to **hourly average arrival delay** across all trips:
  - Group by `date` + `hour` → compute `mean(actual_arrival_delay_min)`.
- Result: a single time series of hourly average delays.
- Fill any missing hours (gaps) using forward-fill or interpolation.
- Save as `output/cleaned/hourly_delay_timeseries.csv`.

### Step 8.2: Normalize the Time Series

- Apply **Min-Max scaling** to [0, 1] range.
- Save `min_delay` and `max_delay` for inverse-transformation later.

### Step 8.3: Create Sequences (Sliding Window)

- Define a **lookback window** of 24 hours (use last 24 hours to predict next hour).
- Create input sequences (X) and target values (y):
  - X[i] = delays at hours [i, i+1, …, i+23]
  - y[i] = delay at hour [i+24]
- Reshape X to 3D array: `(samples, timesteps=24, features=1)`.

### Step 8.4: Train-Test Split

- **80% training / 20% testing** (sequential split, NOT random, to preserve temporal order).
- Log sizes.

### Step 8.5: Build LSTM Architecture

| Layer                  | Configuration                                |
|------------------------|----------------------------------------------|
| LSTM layer 1           | 50 units, `input_shape = c(24, 1)`          |
| Dropout                | rate = 0.2 (regularization)                  |
| Dense layer            | 25 units, ReLU activation                    |
| Output Dense layer     | 1 unit, linear activation (regression)       |

- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)

### Step 8.6: Train the Model

- Epochs: 50 (with early stopping if validation loss plateaus).
- Batch size: 32.
- Validation split: 20% of training data.
- Plot training history (loss over epochs).

### Step 8.7: Evaluate LSTM

- Predict on test set.
- **Inverse-transform** predictions back to original delay scale.
- Compute metrics:

| Metric | Formula                                        | Meaning                         |
|--------|------------------------------------------------|---------------------------------|
| MSE    | mean((actual − predicted)²)                    | Average squared error           |
| RMSE   | √MSE                                          | Error in same units (minutes)   |
| MAE    | mean(\|actual − predicted\|)                   | Average absolute error          |
| MAPE   | mean(\|actual − predicted\| / actual) × 100   | Percentage error                |

### Step 8.8: Visualize LSTM Results

| Plot                               | Type                | Details                                   |
|------------------------------------|---------------------|-------------------------------------------|
| Training loss curve                | Line plot           | Training + validation loss over epochs    |
| Actual vs Predicted (time series)  | Overlaid line plot  | Blue = actual, Red dashed = predicted     |
| Scatter: Actual vs Predicted       | Scatter + 45° line  | Points near diagonal = good predictions   |
| Residual distribution              | Histogram           | Should be centered around 0               |

### Step 8.9: Save LSTM Outputs

- Model → `output/models/lstm_model.h5`
- Normalization params → `output/models/lstm_normalization_params.rds`
- Metrics → `output/reports/lstm_performance_metrics.csv`
- Plots → `output/plots/`

---

## 🌐 PHASE 9: DELAY PROPAGATION NETWORK ANALYSIS
**Script:** `09_propagation.R`

### Step 9.1: Build Station Connectivity Graph

- Each row represents a trip from `origin_station` → `destination_station`.
- Create **edges** between station pairs.
- For each edge (station pair), compute:
  - Number of trips on this connection.
  - Average arrival delay.
  - Average delay change (`delay_change` = arrival delay − departure delay).
  - If delay increases en-route → this connection amplifies delays.

### Step 9.2: Construct igraph Network

- Nodes = unique stations (50 nodes).
- Edges = origin→destination connections, weighted by average delay increase.
- Compute node-level metrics:
  - **Degree centrality:** How many connections does each station have?
  - **PageRank:** Which stations are the most "important" delay propagators?
  - **Betweenness centrality:** Which stations sit on the most delay paths?

### Step 9.3: Identify Top Propagation Bottlenecks

- Rank stations by PageRank score.
- Top 10 stations = primary delay propagation hubs.
- Cross-reference with cluster labels from Phase 6 (are they in the "Bottleneck-Prone" cluster?).

### Step 9.4: Visualize the Network

| Plot                           | Tool              | Details                                        |
|--------------------------------|-------------------|------------------------------------------------|
| Static network graph           | `igraph::plot()`  | Nodes coloured by PageRank, sized by degree    |
| Interactive network            | `networkD3`       | Zoomable, hoverable, saved as HTML             |
| Top propagators bar chart      | `ggplot2`         | PageRank scores for top 10 stations            |

### Step 9.5: Save Network Outputs

- Edge list → `output/reports/delay_propagation_edges.csv`
- Node metrics → `output/reports/station_network_metrics.csv`
- Network plot → `output/plots/delay_propagation_network.png`
- Interactive network → `output/plots/interactive_delay_network.html`

---

## 📊 PHASE 10: VISUALIZATION & POWER BI DASHBOARDS
**Script:** `10_reporting.R`

### Step 10.1: Prepare Power BI Data Exports

Export these CSVs specifically formatted for Power BI import:

| File                              | Contents                                                 |
|-----------------------------------|----------------------------------------------------------|
| `powerbi_overall_summary.csv`     | KPI metrics (total trips, avg delay, delay rate, etc.)   |
| `powerbi_station_rankings.csv`    | Station name, rank, avg delay, delay rate                |
| `powerbi_hourly_patterns.csv`     | Hour, day type, avg delay, delay count                   |
| `powerbi_cluster_summary.csv`     | Cluster label, size, avg delay, description              |
| `powerbi_weather_impact.csv`      | Weather condition, avg delay, delay rate                  |
| `powerbi_model_comparison.csv`    | Model name, Accuracy, F1, AUC                            |
| `powerbi_transformed_full.csv`    | Full cleaned + transformed dataset for slicing           |

### Step 10.2: Create R Combined Dashboard Plot

Create a 2×2 composite plot using `gridExtra::grid.arrange()`:
1. **Top-left:** Delay distribution histogram.
2. **Top-right:** Peak vs off-peak average delay bar.
3. **Bottom-left:** Top 10 delayed stations (horizontal bar).
4. **Bottom-right:** Hourly delay pattern line chart.

Save as `output/plots/comprehensive_delay_dashboard.png` (16×12 inches).

### Step 10.3: Power BI Dashboard Specification

| Dashboard Page           | Components                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| **1. Overview**          | KPI cards, daily delay trend line, delay distribution histogram            |
| **2. Station Analysis**  | Top 20 stations table, station-hour heatmap, cluster pie chart             |
| **3. Weather & Events**  | Weather impact bars, event impact bars, congestion scatter                  |
| **4. Model Performance** | Model comparison bars, ROC curves, confusion matrices, LSTM prediction plot |
| **5. Network**           | Delay propagation visualisation, top propagators table                      |
| **Interactive Filters**  | Date range, transport type, route, station, weather, season slicers        |

---

## 📝 PHASE 11: ERROR ANALYSIS & VALIDATION
**Script:** (integrated into `07_classification.R` and `08_lstm.R`)

### Step 11.1: Classification Error Analysis

- Identify all **false positives** (predicted delayed but was on-time) and **false negatives** (predicted on-time but was delayed).
- Analyse: what features do false negatives share? (e.g., storms with moderate congestion that the model underweighted.)
- Visualize: scatter plot of `avg_delay` vs `traffic_congestion_index` coloured by error type.

### Step 11.2: LSTM Residual Analysis

- Compute residuals = actual − predicted.
- Plot residuals over time → look for systematic over/under-prediction patterns.
- Histogram of residuals → should be approximately normal, centred at 0.
- If residuals show patterns → model is missing something.

### Step 11.3: 10-Fold Cross-Validation

- Run 10-fold CV on the best classification model.
- Report mean and standard deviation of Accuracy, F1, and AUC across folds.
- This demonstrates model stability / robustness.

### Step 11.4: Feature Importance Deep Dive

- From Random Forest: extract and rank all features by importance.
- Report top 10 most important features.
- Discuss: does the ranking align with domain knowledge? (e.g., traffic congestion and weather severity should rank high.)

---

## 📋 PHASE 12: DOCUMENTATION & FINAL REPORT
**Script:** `10_reporting.R`

### Step 12.1: Generate Project Summary

Compile a final project summary covering:

| Section                     | Content                                                          |
|-----------------------------|------------------------------------------------------------------|
| Data Overview               | Dataset description, records, features, date range               |
| Data Cleaning Summary       | Cleaning log with before/after counts                            |
| Feature Engineering         | New features created, encoding strategies                        |
| EDA Key Findings            | Top insights with supporting visualisations                      |
| PCA Results                 | Components selected, variance retained                           |
| Clustering Results          | Number of clusters, profiles, bottleneck identification          |
| Classification Results      | Model comparison table, best model, key metrics                  |
| LSTM Forecasting Results    | Architecture, RMSE, MAE, prediction plots                        |
| Network Analysis            | Top propagator stations, network graph                           |
| Conclusions & Recommendations| Actionable insights for transit operations                      |

### Step 12.2: List All Output Files

Generate a manifest of all outputs:

| Category    | Files                                                                              |
|-------------|------------------------------------------------------------------------------------|
| Cleaned Data| `transport_delays_cleaned.csv`, `transport_delays_transformed.csv`, `pca_transformed_data.csv`, `clustered_data.csv`, `hourly_delay_timeseries.csv` |
| Models      | `pca_model.rds`, `best_classification_model.rds`, `lstm_model.h5`                |
| Plots       | 20+ PNG files covering EDA, PCA, clustering, ML, LSTM, network                    |
| Reports     | `data_profile.csv`, `cleaning_log.csv`, `feature_dictionary.csv`, `station_origin_statistics.csv`, `route_statistics.csv`, `pca_variance_explained.csv`, `cluster_profiles.csv`, `model_comparison.csv`, `lstm_performance_metrics.csv`, `station_network_metrics.csv` |
| Power BI    | 7 CSV exports for Power BI dashboard                                              |

---

## 📌 KEY FORMULAS & METRICS REFERENCE

### Delay Metrics
$$\text{Delay Rate} = \frac{\text{Count of delayed trips}}{\text{Total trips}} \times 100$$

$$\text{Total Delay} = \text{Departure Delay} + \text{Arrival Delay}$$

$$\text{Delay Change} = \text{Arrival Delay} - \text{Departure Delay}$$

### PCA
$$\text{Variance Explained}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j} \times 100$$

### Classification Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### LSTM Regression Metrics
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

---

## 🚀 EXECUTION ORDER & DEPENDENCIES

```
Phase 0  → Environment Setup (run once)
  │
Phase 1  → Data Loading & Profiling
  │
Phase 2  → Data Cleaning ────────────────────────┐
  │                                                │
Phase 3  → Data Transformation & Feature Eng.     │ These three are the
  │                                                │ CORE data mining steps
Phase 4  → Exploratory Data Analysis ─────────────┘
  │
  ├──→ Phase 5  → PCA (Dimensionality Reduction)
  │       │
  │       └──→ Phase 6  → Clustering (Bottleneck Discovery)
  │
  ├──→ Phase 7  → ML Classification (uses transformed data)
  │
  ├──→ Phase 8  → LSTM Forecasting (uses time-series aggregate)
  │
  └──→ Phase 9  → Network Analysis (uses station-pair data)
         │
         └──→ Phase 10 → Dashboards & Visualization
                 │
                 └──→ Phase 11 → Error Analysis & Validation
                         │
                         └──→ Phase 12 → Documentation & Report
```

> **Phases 5–9 can be worked on in parallel** once Phases 1–4 are complete.

---

## ✅ PROJECT COMPLETION CHECKLIST

### Data Mining Core ✓
- [ ] Data cleaning (missing values, duplicates, outliers, invalid data)
- [ ] Data transformation (type conversions, feature engineering, encoding)
- [ ] Data reduction (PCA — dimensionality reduction)
- [ ] Pattern discovery (EDA visualisations, correlation analysis)
- [ ] Clustering (K-Means, DBSCAN — bottleneck identification)

### Machine Learning ✓
- [ ] Logistic Regression (baseline)
- [ ] Random Forest
- [ ] SVM (Support Vector Machine)
- [ ] Model evaluation (Confusion Matrix, ROC, AUC, F1)
- [ ] Cross-validation (10-fold)
- [ ] Feature importance analysis

### Deep Learning ✓
- [ ] LSTM model for hourly delay forecasting
- [ ] Time-series preparation (sequences, normalisation)
- [ ] Training + evaluation (RMSE, MAE, MAPE)
- [ ] Actual vs Predicted visualisation

### Network Analysis ✓
- [ ] Station connectivity graph
- [ ] PageRank / centrality analysis
- [ ] Top delay propagator identification

### Visualisation & Reporting ✓
- [ ] 20+ EDA plots in R (ggplot2)
- [ ] Power BI dashboards (5 pages)
- [ ] Interactive network graph (HTML)
- [ ] Final summary report

---

## 🎓 ACADEMIC REPORT WRITING GUIDE

### Data Cleaning Section
> *"The raw dataset contained 2001 trip records across 24 features. Data cleaning involved: (1) identifying and imputing X missing values using median/mode imputation, (2) removing Y duplicate records, (3) capping Z outliers using the IQR method, and (4) validating logical consistency across fields. The final cleaned dataset retained N records (P% of original)."*

### Data Transformation Section
> *"Feature engineering created M new derived features including temporal features (hour, day type, time-of-day bucket), delay interaction features (total delay, delay change), and composite impact scores (weather severity index, event impact score). Categorical variables were encoded using both label encoding (for clustering) and one-hot encoding (for classification models)."*

### PCA Section
> *"Principal Component Analysis reduced K original numeric features to C principal components, retaining X% of the total variance. The first two components explained Y% of variance, with PC1 primarily driven by delay metrics and PC2 by weather factors."*

### Clustering Section
> *"K-Means clustering with K=3 (determined via elbow and silhouette analysis) identified three distinct trip profiles: On-Time Stable (Cluster 1, N1 trips), Moderately Delayed (Cluster 2, N2 trips), and Bottleneck-Prone (Cluster 3, N3 trips). Bottleneck-prone trips were characterised by high traffic congestion, adverse weather, and peak-hour timing."*

### Classification Section
> *"Three classification algorithms were evaluated for predicting trip delays. Random Forest achieved the highest performance (AUC = X, F1 = Y), outperforming Logistic Regression and SVM. Feature importance analysis revealed traffic congestion index and weather severity as the strongest predictors."*

### LSTM Section
> *"An LSTM neural network with 50 units was trained on hourly aggregated delay data to forecast short-term delays. Using a 24-hour lookback window, the model achieved RMSE of X minutes on the test set, demonstrating effective temporal pattern learning."*

---

**END OF DETAILED IMPLEMENTATION PLAN**

*This plan is tailored specifically to the `public_transport_delays.csv` dataset (2001 records, 24 columns). Follow the phases sequentially from 0 to 12, using the R scripts as outlined. Each phase produces documented outputs that feed into subsequent phases and into the final project report.*
