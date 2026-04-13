# Phase 3: Data Transformation & Feature Engineering (`03_transformation.R`)

## Purpose
The **core feature engineering phase**. Converts cleaned data into a high-dimensional, model-ready dataset. We go from 24 original columns to 60+ columns — each new feature encoding domain knowledge that allows models to detect bottleneck patterns.

---

## Step-by-Step Breakdown

### Step 3.1: Create Datetime Columns
The dataset stores date and time as separate text columns. We merge them into unified `POSIXct` timestamps for arithmetic.

```r
# Combine "2024-01-15" + "08:45:00" → POSIXct datetime object
df$datetime         <- as.POSIXct(paste(df$date, df$time), format = "%Y-%m-%d %H:%M:%S")
df$sched_dep_datetime <- as.POSIXct(paste(df$date, df$scheduled_departure), format = "%Y-%m-%d %H:%M:%S")
df$sched_arr_datetime <- as.POSIXct(paste(df$date, df$scheduled_arrival),  format = "%Y-%m-%d %H:%M:%S")
```

**The Overnight Problem:**
```
Trip: Bus departing at 23:50, scheduled arrival at 00:15 (next day)

Without fix:  arrival (00:15) - departure (23:50) = -23h 35min  ← WRONG
With fix:     arrival + 1 day = 00:15 next day → duration = 25 min  ← CORRECT
```

```r
# Detect overnight: arrival before departure on the same calendar date
overnight_mask <- df$sched_arr_datetime < df$sched_dep_datetime
df$sched_arr_datetime[overnight_mask] <- df$sched_arr_datetime[overnight_mask] + days(1)

cat(sprintf("Overnight trips adjusted: %d\n", sum(overnight_mask, na.rm = TRUE)))
# Example output: "Overnight trips adjusted: 67"
```

---

### Step 3.2: Extract Temporal Features
Time-based features are critical because transport follows a cyclical pattern — Monday rush hour at 8 AM is very different from Saturday noon.

```r
df$hour         <- hour(df$datetime)      # 0–23
df$day_name     <- weekdays(df$date)      # "Monday", "Tuesday", etc.
df$is_weekend   <- ifelse(df$weekday %in% c(0, 6), 1, 0)  # 0=weekday, 1=weekend
df$month        <- month(df$date)         # 1–12
df$week_of_year <- week(df$date)          # 1–52
```

**Time-of-Day Bucket:**
```r
df$time_of_day_bucket <- case_when(
  df$hour >= 0  & df$hour < 6  ~ "Night",      # 00:00–05:59 (low traffic)
  df$hour >= 6  & df$hour < 12 ~ "Morning",    # 06:00–11:59 (morning rush)
  df$hour >= 12 & df$hour < 18 ~ "Afternoon",  # 12:00–17:59 (stable)
  df$hour >= 18 & df$hour <= 23 ~ "Evening",   # 18:00–23:59 (evening rush)
  TRUE ~ "Unknown"
)
```

**Example:** A trip recorded at `datetime = 2024-03-12 08:23:00` gets:
`hour = 8`, `day_name = "Tuesday"`, `is_weekend = 0`, `time_of_day_bucket = "Morning"`

---

### Step 3.3: Compute Trip-Level Delay Features
Rather than just having raw delay values, we engineer features that capture *delay dynamics* — how delays behave and change.

```r
# Total delay: combined departure + arrival delay
df$total_delay  <- df$actual_departure_delay_min + df$actual_arrival_delay_min

# Delay change: did the delay grow or shrink during the trip?
df$delay_change <- df$actual_arrival_delay_min - df$actual_departure_delay_min
```

**Interpretable Examples:**
```
Trip A: dep_delay=+5 min, arr_delay=+12 min → delay_change = +7  (delay GREW en-route, route bottleneck)
Trip B: dep_delay=+8 min, arr_delay=+3 min  → delay_change = -5  (driver RECOVERED time, efficient route)
Trip C: dep_delay=0 min,  arr_delay=+18 min → delay_change = +18 (incident happened mid-route)
```

```r
# Early flags: negative delay = arrived/departed BEFORE schedule
df$is_early_departure <- ifelse(df$actual_departure_delay_min < 0, 1, 0)
df$is_early_arrival   <- ifelse(df$actual_arrival_delay_min   < 0, 1, 0)

# Severe delay flag: a binary target for extreme delay classification
df$severe_delay <- ifelse(df$actual_arrival_delay_min > 15, 1, 0)
```

---

### Step 3.4: Encode Categorical Variables
Machine learning models require numbers, not text. We encode categories as ordered integers where the ordering encodes domain knowledge.

```r
# Transport types — encoded by operational complexity
transport_map <- c("Bus" = 1, "Metro" = 2, "Train" = 3, "Tram" = 4)
df$transport_type_enc <- transport_map[as.character(df$transport_type)]

# Weather — encoded by SEVERITY (not alphabetically!)
weather_map <- c("Clear" = 1, "Cloudy" = 2, "Fog" = 3, "Rain" = 4, "Snow" = 5, "Storm" = 6)
df$weather_condition_enc <- weather_map[as.character(df$weather_condition)]
```

**Why severity ordering for weather?**
If we used alphabetical encoding (`Clear=1, Cloudy=2, Fog=3, Rain=4, Snow=5, Storm=6`), the model gets the same result. But any label assignment preserves relative severity for ordinal regression. The model can learn "higher number = worse weather".

**Example:** Row with `weather_condition = "Storm"` gets `weather_condition_enc = 6`. A row with `"Clear"` gets `1`. The model can now reason numerically about weather severity.

---

### Step 3.5: Interaction & Derived Weather Features
Raw features can miss multiplicative effects. A 20 km/h wind by itself doesn't delay buses. But 20 km/h wind + heavy rain + storm = major delays. We capture this explicitly.

```r
# Storm indicator: binary flag for the worst weather types
storm_indicator <- ifelse(as.character(df$weather_condition) %in% c("Storm", "Snow"), 1, 0)

# Weather Severity Index: weighted composite score
#   Wind speed: 0.3 weight (lower — wind alone is manageable)
#   Precipitation: 0.4 weight (highest — rain has biggest operational impact)
#   Storm binary: 0.3 weight (extreme condition indicator)
df$weather_severity_index <- round(
  0.3 * (df$wind_speed_kmh / max(df$wind_speed_kmh, na.rm = TRUE)) +
  0.4 * (df$precipitation_mm / max(df$precipitation_mm + 0.01, na.rm = TRUE)) +
  0.3 * storm_indicator,
  4
)
```

**Concrete Example:**
```
Wind: 80 km/h → 0.3 × (80/120) = 0.200
Rain: 40 mm   → 0.4 × (40/85)  = 0.188
Storm: YES    → 0.3 × 1        = 0.300
weather_severity_index = 0.688   ← High severity
```

```r
# Congestion × Weather: captures the "double trouble" effect
df$congestion_weather_combo <- round(df$traffic_congestion_index * df$weather_severity_index, 4)
```

**Example:** Traffic index 0.8 during storm severity 0.7 → combo = 0.56, much higher than either alone.

---

### Step 3.6–3.8: Historical Aggregation (Station, Route, Hour)
Every trip gets a new feature: "what is the historical average delay at this station / on this route?"

```r
station_stats <- df %>%
  group_by(origin_station) %>%
  summarise(
    station_avg_dep_delay = round(mean(actual_departure_delay_min, na.rm = TRUE), 2),
    station_delay_rate    = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    station_max_delay     = max(actual_arrival_delay_min, na.rm = TRUE),
    .groups = "drop"
  )

# Merge back so every trip "knows" its station's history
df <- df %>% left_join(station_stats %>% select(origin_station, station_avg_dep_delay,
                                                  station_delay_rate), by = "origin_station")
```

**Example:** Trip from `Station_07`:
- `station_avg_dep_delay = 8.3 min` → This station habitually departs late
- `station_delay_rate = 62%` → 62% of trips from here are delayed
- Now the model has historical context without "peeking" at the current trip's own outcome

---

### Step 3.9: Dual Normalization
Prepares data for two completely different algorithm families simultaneously.

```r
for (col in scale_cols) {
  col_mean <- mean(as.numeric(df[[col]]), na.rm = TRUE)
  col_sd   <- sd(as.numeric(df[[col]]), na.rm = TRUE)

  # Z-Score: centers at 0, scales to unit variance (for PCA, clustering, XGBoost)
  if (col_sd > 0) {
    df[[paste0(col, "_z")]] <- round((as.numeric(df[[col]]) - col_mean) / col_sd, 4)
  } else {
    df[[paste0(col, "_z")]] <- 0  # Constant column: no variance, set to 0
  }
}

for (col in scale_cols) {
  col_min <- min(as.numeric(df[[col]]), na.rm = TRUE)
  col_max <- max(as.numeric(df[[col]]), na.rm = TRUE)

  # Min-Max: squashes to [0, 1] range (for LSTM neural networks)
  if (range_val > 0) {
    df[[paste0(col, "_mm")]] <- round((as.numeric(df[[col]]) - col_min) / range_val, 4)
  } else {
    df[[paste0(col, "_mm")]] <- 0
  }
}
```

**Concrete Example for `actual_arrival_delay_min`:**
```
Values: [-5, 0, 4, 12, 27]   mean=7.6, SD=11.2,  min=-5, max=27

Z-Score  : [-1.12, -0.68, -0.32, 0.39, 1.73]   → centred around 0
Min-Max  : [0.00,  0.16,  0.28, 0.53, 1.00]    → compressed to [0, 1]
```

---

## Outputs
| File | Description |
|---|---|
| `output/cleaned/transport_delays_transformed.rds` | Feature-engineered dataset (60+ columns) |
| `output/reports/feature_dictionary.csv` | Maps every column to its origin method |
| `output/reports/scaling_parameters.csv` | Mean, SD, Min, Max for each scaled column |
| `output/reports/station_origin_statistics.csv` | Historical delay stats per station |
| `output/reports/route_statistics.csv` | Historical delay stats per route |

---

## 💡 Presentation Talking Points
> "When we created `station_delay_rate`, every trip gained a historical context score for its station — without looking at its own delay. A model trained on this can immediately flag that trips from Station_07, which has a 62% delay rate, are high-risk — even before any weather or traffic data is considered."

> "Our dual normalization approach — Z-Score and Min-Max — means we only run Phase 3 once but produce data ready for both traditional algorithms (PCA, K-Means) and deep learning (LSTM)."
