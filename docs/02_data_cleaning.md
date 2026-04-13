# Phase 2: Data Cleaning (`02_data_cleaning.R`)

## Purpose
Transforms the raw dataset into a **reliable, analysis-ready** form. Every action taken is logged to a `cleaning_log` for full traceability — so we can always answer: "What did you change, why, and how many records were affected?"

---

## The Audit Trail Pattern
Before any cleaning begins, a `cleaning_log` is initialised and updated after every step. This is a professional-grade practice for reproducible data science.

```r
cleaning_log <- data.frame(
  Step = character(), Records_Before = integer(),
  Records_After = integer(), Records_Changed = integer(),
  Method = character(), stringsAsFactors = FALSE
)

add_log <- function(step, before, after, method) {
  cleaning_log <<- bind_rows(cleaning_log, data.frame(
    Step            = step,
    Records_Before  = before,
    Records_After   = after,
    Records_Changed = before - after,   # Negative = records were added (imputed)
    Method          = method
  ))
}
```

**Example final log output:**
```
                    Step  Records_Before  Records_After  Records_Changed  Method
          Raw data loaded           10000          10000                0       —
  Missing values handled            10000           9987               13  Median/Mode + Drop
       Duplicates removed            9987           9975               12  distinct() + slice(1)
          Outliers capped            9975           9975                0  IQR winsorization
 Logical consistency OK             9975           9975                0  Flag circular trips
```

---

## Step-by-Step Breakdown

### Step 2.1: Handle Missing Values
Different columns require different strategies based on their role in the data.

**Strategy 1 — Drop the row (critical key columns):**
```r
# trip_id is a primary key — you can't meaningfully impute a unique identifier
if (sum(is.na(df$trip_id)) > 0) {
  df <- df %>% filter(!is.na(trip_id))
}

# Both delay columns NA = the row has no analytical value at all
both_na <- is.na(df$actual_departure_delay_min) & is.na(df$actual_arrival_delay_min)
df <- df %>% filter(!both_na)
```

**Strategy 2 — Median imputation (numeric columns):**
Median is preferred over mean for weather/traffic variables because they tend to be right-skewed (occasional storms, traffic events create high outliers).

```r
for (col in c("temperature_C", "humidity_percent", "wind_speed_kmh",
              "precipitation_mm", "event_attendance_est", "traffic_congestion_index")) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    med_val <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med_val
    cat(sprintf("  Imputed %d NAs in '%s' with median = %.2f\n", na_count, col, med_val))
  }
}
```

**Example output:**
```
Imputed 87 NAs in 'temperature_C' with median = 15.30
Imputed 34 NAs in 'wind_speed_kmh' with median = 18.20
```

**Strategy 3 — Mode imputation (categorical columns):**
A custom `get_mode()` function finds the most frequent value, which is the best guess for an unknown category.

```r
get_mode <- function(x) {
  ux <- na.omit(x)
  ux[which.max(tabulate(match(x, ux)))]  # Returns the most frequent unique value
}

for (col in c("transport_type", "weather_condition", "event_type", "season")) {
  mode_val <- get_mode(df[[col]])
  df[[col]][is.na(df[[col]])] <- mode_val
}
```

**Example:** If `weather_condition` is missing for 50 rows and "Clear" appears 4800 times (most frequently), all 50 NAs become "Clear" — the statistically most probable value.

---

### Step 2.2: Remove Duplicates
Two levels of deduplication to catch different types of duplicates.

```r
# Level 1: Exact full-row copies (all columns identical)
full_dups <- sum(duplicated(df))
df <- df %>% distinct()

# Level 2: Logical duplicates — same trip_id appearing more than once
# This happens when the same trip is recorded twice with slightly different NA patterns
trip_dup_ids <- df$trip_id[duplicated(df$trip_id)]
df <- df %>% group_by(trip_id) %>% slice(1) %>% ungroup()
```

**Example:** `trip_id = "TRP_5541"` appears twice — once with `weather_condition = "Rain"` and once with `weather_condition = NA` (data pipeline glitch). We keep the first occurrence.

---

### Step 2.3: Validate & Correct Data Types
Ensures every column is in the type expected by downstream scripts.

```r
# Categoricals become factors (enables proper grouping and frequency tables)
factor_cols <- c("transport_type", "weather_condition", "event_type",
                 "season", "route_id", "origin_station", "destination_station")
for (col in factor_cols) df[[col]] <- as.factor(df[[col]])

# Binary columns become factors for classification algorithms
df$holiday   <- as.factor(df$holiday)  # 0/1 → "0"/"1" factor
df$peak_hour <- as.factor(df$peak_hour)
df$delayed   <- as.factor(df$delayed)  # Our ML target variable
```

**Why factor vs character?** Factors store the set of valid levels, enabling better memory usage and preventing invalid values from sneaking in. An analysis of `weather_condition` as a character could accidentally include typos like `"Rainn"` as a valid level.

---

### Step 2.4: Outlier Handling — IQR Winsorization
We **cap** outliers (winsorize) rather than delete them. This preserves every row but limits extreme values' influence on models.

```r
for (col in outlier_cols) {
  vals <- df[[col]]
  Q1      <- quantile(vals, 0.25, na.rm = TRUE)   # 25th percentile
  Q3      <- quantile(vals, 0.75, na.rm = TRUE)   # 75th percentile
  IQR_val <- Q3 - Q1                              # Interquartile range

  lower <- Q1 - 1.5 * IQR_val   # Standard outlier boundary
  upper <- Q3 + 1.5 * IQR_val

  # pmin: cap from above (nothing exceeds `upper`)
  # pmax: cap from below (nothing goes below `lower`)
  df[[col]] <- pmax(pmin(vals, upper), lower)
}
```

**Concrete Example for `actual_arrival_delay_min`:**
```
Q1 = 2 min,  Q3 = 12 min,  IQR = 10 min
Lower bound = 2 - 15 = -13 min   (early arrivals capped at 13 min early)
Upper bound = 12 + 15 = 27 min   (delays capped at 27 min)

A recorded delay of 180 min (sensor error / outlier) → capped to 27 min
A recorded early arrival of -45 min (impossible) → capped to -13 min
```

---

### Step 2.5: Logical Consistency Validation
Checks that the data makes sense *as transport data*, not just as numbers.

```r
# Circular trips: a vehicle cannot depart and arrive at the same station
circular <- sum(df$origin_station == df$destination_station, na.rm = TRUE)
if (circular > 0) {
  df$is_circular_trip <- ifelse(df$origin_station == df$destination_station, 1, 0)
  # We FLAG, not delete — they may be valid loop routes worth investigating
}

# Potential label error: high delay but marked "not delayed"
# The `delayed` flag might use a different threshold (e.g., >5 min) than our data
high_dep_not_delayed <- sum(df$actual_departure_delay_min > 10 & df$delayed == "0")
cat(sprintf("  High dep delay (>10 min) but delayed=0: %d\n", high_dep_not_delayed))
```

**Example output:**
```
Circular trips: 12    ← Flagged with is_circular_trip = 1
High dep delay (>10 min) but delayed=0: 234   ← Threshold mismatch: they use 15 min
Possible overnight trips (arrival < departure): 67  ← Handled in Phase 3
```

---

## Outputs
| File | Description |
|---|---|
| `output/reports/missing_value_summary.csv` | Before/after NA counts per column |
| `output/reports/outlier_detection_log.csv` | Outlier bounds, counts per column |
| `output/reports/cleaning_log.csv` | Full audit trail of all cleaning steps |
| `output/cleaned/transport_delays_cleaned.csv` | Final clean dataset as CSV |
| `output/cleaned/transport_delays_cleaned.rds` | Final clean dataset as RDS (for Phase 3) |

---

## 💡 Presentation Talking Points
> "We chose winsorization (capping) over deletion for outliers. A recorded 180-minute delay might be a sensor error, but it might also be a real major incident. Capping it at the 1.5×IQR bound preserves the row and its other features while limiting the extreme value's distortion."

> "Our cleaning log gives us a complete audit trail — any reviewer can see that we went from 10,000 to 9,975 rows, and exactly why each row was removed."
