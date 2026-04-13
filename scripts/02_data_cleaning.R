# =============================================================================
# PHASE 2: DATA CLEANING (CRITICAL PHASE)
# Script: 02_data_cleaning.R
# Purpose: Handle missing values, remove duplicates, correct data types,
#          detect and handle outliers, validate logical consistency.
# =============================================================================

cat("=== PHASE 2: DATA CLEANING ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)

# --- Load raw data from Phase 1 ----------------------------------------------
df <- readRDS("output/cleaned/raw_loaded.rds")
initial_rows <- nrow(df)
cat(sprintf("Loaded %d rows x %d columns\n\n", nrow(df), ncol(df)))

# Initialise cleaning log
cleaning_log <- data.frame(
  Step            = character(),
  Records_Before  = integer(),
  Records_After   = integer(),
  Records_Changed = integer(),
  Method          = character(),
  stringsAsFactors = FALSE
)

add_log <- function(step, before, after, method) {
  cleaning_log <<- bind_rows(cleaning_log, data.frame(
    Step            = step,
    Records_Before  = before,
    Records_After   = after,
    Records_Changed = before - after,
    Method          = method,
    stringsAsFactors = FALSE
  ))
}

add_log("Raw data loaded", initial_rows, initial_rows, "—")

# =============================================================================
# Step 2.1: Handle Missing Values
# =============================================================================
cat("Step 2.1: Handling missing values...\n")

missing_summary <- data.frame(
  Column          = names(df),
  Missing_Before  = colSums(is.na(df)),
  stringsAsFactors = FALSE
)

before_rows <- nrow(df)

# --- Drop rows where trip_id is NA (critical key) ---
if (sum(is.na(df$trip_id)) > 0) {
  df <- df %>% filter(!is.na(trip_id))
  cat(sprintf("  Dropped %d rows with missing trip_id\n", before_rows - nrow(df)))
}

# --- Drop rows where BOTH delay columns are NA ---
both_na <- is.na(df$actual_departure_delay_min) & is.na(df$actual_arrival_delay_min)
if (sum(both_na) > 0) {
  df <- df %>% filter(!both_na)
  cat(sprintf("  Dropped %d rows with both delay columns NA\n", sum(both_na)))
}

# --- Impute numeric weather columns with median ---
numeric_impute_cols <- c("temperature_C", "humidity_percent", "wind_speed_kmh",
                         "precipitation_mm", "event_attendance_est",
                         "traffic_congestion_index")

for (col in numeric_impute_cols) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    med_val <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med_val
    cat(sprintf("  Imputed %d missing values in '%s' with median = %.2f\n",
                na_count, col, med_val))
  }
}

# --- Impute delay columns individually with median ---
for (col in c("actual_departure_delay_min", "actual_arrival_delay_min")) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    med_val <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med_val
    cat(sprintf("  Imputed %d missing values in '%s' with median = %.2f\n",
                na_count, col, med_val))
  }
}

# --- Impute categorical columns with mode ---
get_mode <- function(x) {
  ux <- na.omit(x)
  ux[which.max(tabulate(match(x, ux)))]
}

cat_impute_cols <- c("transport_type", "weather_condition", "event_type",
                     "season", "route_id", "origin_station", "destination_station")

for (col in cat_impute_cols) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    mode_val <- get_mode(df[[col]])
    df[[col]][is.na(df[[col]])] <- mode_val
    cat(sprintf("  Imputed %d missing values in '%s' with mode = '%s'\n",
                na_count, col, mode_val))
  }
}

# --- Impute binary flags with mode ---
binary_cols <- c("holiday", "peak_hour", "delayed")
for (col in binary_cols) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    mode_val <- get_mode(df[[col]])
    df[[col]][is.na(df[[col]])] <- mode_val
    cat(sprintf("  Imputed %d missing values in '%s' with mode = %s\n",
                na_count, col, mode_val))
  }
}

# --- Impute weekday with mode ---
if (sum(is.na(df$weekday)) > 0) {
  mode_val <- get_mode(df$weekday)
  df$weekday[is.na(df$weekday)] <- mode_val
}

# --- Impute date and time columns ---
if (sum(is.na(df$date)) > 0) {
  df <- df %>% filter(!is.na(date))
  cat(sprintf("  Dropped rows with missing date\n"))
}

for (col in c("time", "scheduled_departure", "scheduled_arrival")) {
  na_count <- sum(is.na(df[[col]]))
  if (na_count > 0) {
    mode_val <- get_mode(df[[col]])
    df[[col]][is.na(df[[col]])] <- mode_val
    cat(sprintf("  Imputed %d missing values in '%s' with mode\n", na_count, col))
  }
}

# Verify no NA remains
remaining_na <- colSums(is.na(df))
if (sum(remaining_na) == 0) {
  cat("  ✓ All missing values handled — zero NAs remaining.\n")
} else {
  cat("  WARNING: Remaining NAs:\n")
  print(remaining_na[remaining_na > 0])
}

missing_summary$Missing_After <- colSums(is.na(df[, missing_summary$Column]))
write_csv(missing_summary, "output/reports/missing_value_summary.csv")
cat("  Saved: output/reports/missing_value_summary.csv\n")

add_log("Missing values handled", before_rows, nrow(df), "Median/Mode imputation + Drop")

# =============================================================================
# Step 2.2: Remove Duplicate Records
# =============================================================================
cat("\nStep 2.2: Removing duplicates...\n")

before_rows <- nrow(df)

# Full-row duplicates
full_dups <- sum(duplicated(df))
if (full_dups > 0) {
  df <- df %>% distinct()
  cat(sprintf("  Removed %d full-row duplicates\n", full_dups))
}

# Logical duplicates: same trip_id appearing more than once
trip_dup_ids <- df$trip_id[duplicated(df$trip_id)]
if (length(trip_dup_ids) > 0) {
  df <- df %>% group_by(trip_id) %>% slice(1) %>% ungroup()
  cat(sprintf("  Removed %d duplicate trip_id entries (kept first occurrence)\n",
              length(trip_dup_ids)))
} else {
  cat("  No duplicate trip_ids found.\n")
}

cat(sprintf("  Duplicates check: %d full-row, %d trip_id duplicates\n",
            full_dups, length(trip_dup_ids)))

add_log("Duplicates removed", before_rows, nrow(df), "distinct() + first occurrence")

# =============================================================================
# Step 2.3: Validate & Correct Data Types
# =============================================================================
cat("\nStep 2.3: Validating and correcting data types...\n")

# Ensure date column is Date type
df$date <- as.Date(df$date)

# Ensure character time columns remain as character (will parse in transformation phase)
df$time                <- as.character(df$time)
df$scheduled_departure <- as.character(df$scheduled_departure)
df$scheduled_arrival   <- as.character(df$scheduled_arrival)

# Convert categoricals to factor
factor_cols <- c("transport_type", "weather_condition", "event_type",
                 "season", "route_id", "origin_station", "destination_station")
for (col in factor_cols) {
  df[[col]] <- as.factor(df[[col]])
}

# Convert binary columns to factor for classification
df$holiday   <- as.factor(df$holiday)
df$peak_hour <- as.factor(df$peak_hour)
df$delayed   <- as.factor(df$delayed)

# Verify all numeric columns are numeric
numeric_check_cols <- c("actual_departure_delay_min", "actual_arrival_delay_min",
                        "temperature_C", "humidity_percent", "wind_speed_kmh",
                        "precipitation_mm", "event_attendance_est",
                        "traffic_congestion_index", "weekday")
for (col in numeric_check_cols) {
  if (!is.numeric(df[[col]])) {
    df[[col]] <- as.numeric(df[[col]])
    cat(sprintf("  Converted '%s' to numeric\n", col))
  }
}

cat("  ✓ Data types validated and corrected.\n")
cat("  Column types after correction:\n")
type_summary <- data.frame(
  Column = names(df),
  Type   = sapply(df, class),
  stringsAsFactors = FALSE
)
print(type_summary, row.names = FALSE)

# =============================================================================
# Step 2.4: Detect and Handle Outliers
# =============================================================================
cat("\nStep 2.4: Detecting and handling outliers (IQR winsorization)...\n")

outlier_cols <- c("actual_departure_delay_min", "actual_arrival_delay_min",
                  "temperature_C", "humidity_percent", "wind_speed_kmh",
                  "precipitation_mm", "traffic_congestion_index",
                  "event_attendance_est")

outlier_log <- data.frame(
  Column           = character(),
  Outliers_Detected = integer(),
  Lower_Bound      = numeric(),
  Upper_Bound      = numeric(),
  stringsAsFactors = FALSE
)

for (col in outlier_cols) {
  vals <- df[[col]]
  Q1   <- quantile(vals, 0.25, na.rm = TRUE)
  Q3   <- quantile(vals, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  
  outlier_count <- sum(vals < lower | vals > upper, na.rm = TRUE)
  
  # Winsorize: cap at bounds
  df[[col]] <- pmax(pmin(vals, upper), lower)
  
  outlier_log <- bind_rows(outlier_log, data.frame(
    Column            = col,
    Outliers_Detected = outlier_count,
    Lower_Bound       = round(lower, 2),
    Upper_Bound       = round(upper, 2),
    stringsAsFactors  = FALSE
  ))
  
  if (outlier_count > 0) {
    cat(sprintf("  %s: %d outliers capped [%.2f, %.2f]\n",
                col, outlier_count, lower, upper))
  }
}

write_csv(outlier_log, "output/reports/outlier_detection_log.csv")
cat("  Saved: output/reports/outlier_detection_log.csv\n")
add_log("Outliers capped", nrow(df), nrow(df), "IQR winsorization")

# =============================================================================
# Step 2.5: Validate Logical Consistency
# =============================================================================
cat("\nStep 2.5: Validating logical consistency...\n")

# Check: origin_station == destination_station
circular <- sum(df$origin_station == df$destination_station, na.rm = TRUE)
cat(sprintf("  Circular trips (origin == destination): %d\n", circular))
if (circular > 0) {
  # Flag but don't remove — just mark as suspicious
  df$is_circular_trip <- ifelse(df$origin_station == df$destination_station, 1, 0)
  cat("  Flagged circular trips (is_circular_trip column added).\n")
} else {
  df$is_circular_trip <- 0
}

# Check: high departure delay but delayed == 0
high_dep_not_delayed <- sum(
  df$actual_departure_delay_min > 10 & df$delayed == "0", na.rm = TRUE
)
cat(sprintf("  High dep delay (>10 min) but delayed=0: %d (may indicate labelling threshold)\n",
            high_dep_not_delayed))

# Check: scheduled_arrival before scheduled_departure (potential overnight trips)
dep_times <- hms::as_hms(df$scheduled_departure)
arr_times <- hms::as_hms(df$scheduled_arrival)
overnight <- sum(arr_times < dep_times, na.rm = TRUE)
cat(sprintf("  Possible overnight trips (arrival < departure time): %d\n", overnight))

# Log the consistency checks
add_log("Logical consistency validated", nrow(df), nrow(df), "Flag circular trips")

# =============================================================================
# Step 2.6: Save Cleaned Data
# =============================================================================
cat("\nStep 2.6: Saving cleaned data...\n")

write_csv(df, "output/cleaned/transport_delays_cleaned.csv")
cat("  Saved: output/cleaned/transport_delays_cleaned.csv\n")

saveRDS(df, "output/cleaned/transport_delays_cleaned.rds")
cat("  Saved: output/cleaned/transport_delays_cleaned.rds\n")

# Final cleaning log entry
add_log("Final clean dataset", initial_rows, nrow(df), "All above steps combined")

write_csv(cleaning_log, "output/reports/cleaning_log.csv")
cat("  Saved: output/reports/cleaning_log.csv\n")

cat("\n  Cleaning Log Summary:\n")
print(cleaning_log, row.names = FALSE)

cat(sprintf("\n  Final dataset: %d rows (%.1f%% of original %d rows)\n",
            nrow(df), 100 * nrow(df) / initial_rows, initial_rows))

cat("\n=== PHASE 2 COMPLETE ===\n")
