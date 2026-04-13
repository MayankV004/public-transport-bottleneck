# Phase 1: Data Loading & Profiling (`01_load_and_profile.R`)

## Purpose
The **entry point** of the analysis pipeline. Loads the raw dataset and generates a comprehensive data profiling report so we understand the dataset's structure, quality, and distributions *before* touching anything. Think of it as a doctor's first examination before prescribing treatment.

---

## Data Flow

```
data/public_transport_delays.csv
          ↓
    read_csv() with typed schema
          ↓
  Structure inspection (str, glimpse, summary)
  Missing Value Analysis → per-column NA counts
  Unique Value Analysis  → per-column cardinality
  Numeric Statistics     → min/max/mean/median/SD
  Categorical Frequencies → counts per category
  Target Distribution    → delayed=0 vs delayed=1
  Sanity Checks          → circular trips, negative delays
          ↓
output/reports/data_profile.csv
output/cleaned/raw_loaded.rds
```

---

## Step-by-Step Breakdown

### Step 1.1: Typed Schema Loading
We explicitly define the data type for key columns at load time. This prevents silent coercion errors that are hard to debug later.

```r
df <- read_csv(
  "data/public_transport_delays.csv",
  col_types = cols(
    trip_id  = col_character(),   # Force string — never a number
    route_id = col_character(),   # "Route 12" should not become integer 12
    date     = col_date(format = "%Y-%m-%d"),  # Parse "2024-01-15" as a proper Date
    .default = col_guess()        # Let R infer everything else
  ),
  show_col_types = FALSE
)
```

**Example of what can go wrong without this:** If `route_id` contains values like `"007"` and R auto-detects it as numeric, the leading zero is silently dropped, turning `"007"` into `7`. Joins on `route_id` would then fail.

---

### Step 1.2: Structure Inspection
We use three complementary views of the data to understand its shape.

```r
str(df)      # Compact view: column names, types, first few values
glimpse(df)  # Wider view: same info in a more readable format
summary(df)  # Statistical summary for numeric; count table for factors
```

**Example `str()` output snippet:**
```
$ trip_id                   : chr "TRP_001" "TRP_002" ...
$ actual_arrival_delay_min  : num 4.2 -1.0 12.5 0.0 ...
$ weather_condition         : chr "Rain" "Clear" "Snow" ...
```

---

### Step 1.3: Missing Values Per Column
We compute both the raw count and percentage of NAs for every column. This directly informs Phase 2's imputation strategy.

```r
missing_counts <- colSums(is.na(df))
missing_pct    <- round(100 * missing_counts / nrow(df), 2)
```

**Example output:**
```
           Column  Missing_Count  Missing_Percent
   temperature_C             87             0.87
wind_speed_kmh               34             0.34
      event_type              0             0.00
```
This tells us `temperature_C` has ~1% missing — safe to impute with median. If it were 40%, we'd need a different strategy.

---

### Step 1.4: Unique Values Per Column
High cardinality (many unique values) vs low cardinality reveals categorical vs continuous nature of columns.

```r
unique_counts <- sapply(df, n_distinct)
```

**Example output:**
```
      trip_id: 10000   ← Primary key (as expected)
     route_id:    45   ← 45 distinct routes
   season:          4  ← Winter/Spring/Summer/Autumn
   delayed:         2  ← Binary target (0 or 1)
```

---

### Step 1.5: Numeric Statistics (Pivot Pattern)
A complex but elegant `pivot_longer/pivot_wider` pattern computes 5 statistics for every numeric column simultaneously.

```r
numeric_stats <- numeric_cols %>%
  summarise(across(everything(), list(
    min    = ~min(., na.rm = TRUE),
    max    = ~max(., na.rm = TRUE),
    mean   = ~round(mean(., na.rm = TRUE), 2),
    median = ~round(median(., na.rm = TRUE), 2),
    sd     = ~round(sd(., na.rm = TRUE), 2)
  ))) %>%
  pivot_longer(everything(),
               names_to  = c("Column", "Stat"),
               names_sep = "_(?=[^_]+$)",  # Split at the LAST underscore
               values_to = "Value") %>%
  pivot_wider(names_from = Stat, values_from = Value)
```

**Example output table:**
```
                     Column    min    max   mean  median    sd
actual_arrival_delay_min    -5.0   45.0   6.23    4.80  7.12
          temperature_C    -12.0   38.0  14.50   15.00  8.90
      precipitation_mm       0.0   85.0   4.20    0.00 10.30
```

---

### Step 1.6: Target Variable Distribution
Checks class balance — a critical signal for whether we need up/down-sampling in Phase 7.

```r
delay_table <- table(df$delayed)
delay_prop  <- round(prop.table(delay_table) * 100, 2)
```

**Example output:**
```
delayed:  0: 6523  1: 3477    →  65.2% On-Time  |  34.8% Delayed
```
34.8% is close enough to 50/50 that no aggressive resampling is needed — this is surfaced *now* so the ML phase is informed.

---

### Step 1.7: Domain Sanity Checks
Validates transport-specific logic that pure statistics won't catch.

```r
# Circular trips: impossible in real networks, likely a data entry error
circular <- sum(df$origin_station == df$destination_station, na.rm = TRUE)
cat(sprintf("Circular trips (origin == destination): %d\n", circular))

# Negative delays are VALID — they mean the vehicle departed/arrived early
neg_dep <- sum(df$actual_departure_delay_min < 0, na.rm = TRUE)
cat(sprintf("Early departures (negative dep delay): %d\n", neg_dep))
```

**Example output:**
```
Circular trips (origin == destination): 12   ← Flag these for Phase 2
Early departures (negative dep delay): 847   ← Normal: vehicles run ahead sometimes
```

---

## Outputs
| File | Description |
|---|---|
| `output/reports/data_profile.csv` | Combined missing + unique value statistics per column |
| `output/cleaned/raw_loaded.rds` | Raw data saved as RDS (faster loading in Phase 2) |

---

## 💡 Presentation Talking Points
> "The profiling phase told us our target variable had a 35/65 class split — close enough to balanced that we could proceed without aggressive oversampling, but still worth monitoring in model evaluation."

> "We discovered 12 circular trips where origin equals destination. These are logically impossible in a normal transit scenario — we flagged them for investigation in Phase 2, rather than silently dropping them."
