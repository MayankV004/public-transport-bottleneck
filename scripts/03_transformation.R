# =============================================================================
# PHASE 3: DATA TRANSFORMATION & FEATURE ENGINEERING (CRITICAL PHASE)
# Script: 03_transformation.R
# Purpose: Create datetime columns, extract temporal features, compute
#          trip-level delay features, encode categoricals, create interaction
#          features, aggregate by station/route/hour, and normalize.
# =============================================================================

cat("=== PHASE 3: DATA TRANSFORMATION & FEATURE ENGINEERING ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(hms)

# --- Load cleaned data from Phase 2 ------------------------------------------
df <- readRDS("output/cleaned/transport_delays_cleaned.rds")
cat(sprintf("Loaded cleaned data: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 3.1: Create Datetime Columns
# =============================================================================
cat("Step 3.1: Creating datetime columns...\n")

# Combine date + time → datetime (POSIXct)
df$datetime <- as.POSIXct(paste(df$date, df$time), format = "%Y-%m-%d %H:%M:%S")

# Combine date + scheduled_departure → sched_dep_datetime
df$sched_dep_datetime <- as.POSIXct(paste(df$date, df$scheduled_departure),
                                     format = "%Y-%m-%d %H:%M:%S")

# Combine date + scheduled_arrival → sched_arr_datetime
df$sched_arr_datetime <- as.POSIXct(paste(df$date, df$scheduled_arrival),
                                     format = "%Y-%m-%d %H:%M:%S")

# Handle overnight trips: if arrival < departure, add 1 day
overnight_mask <- !is.na(df$sched_arr_datetime) & !is.na(df$sched_dep_datetime) &
  df$sched_arr_datetime < df$sched_dep_datetime
df$sched_arr_datetime[overnight_mask] <- df$sched_arr_datetime[overnight_mask] + days(1)

overnight_count <- sum(overnight_mask, na.rm = TRUE)
cat(sprintf("  Datetime columns created. Overnight trips adjusted: %d\n", overnight_count))

# =============================================================================
# Step 3.2: Extract Temporal Features
# =============================================================================
cat("\nStep 3.2: Extracting temporal features...\n")

df$hour         <- hour(df$datetime)
df$day_name     <- weekdays(df$date)
df$is_weekend   <- ifelse(df$weekday %in% c(0, 6), 1, 0)
df$month        <- month(df$date)
df$week_of_year <- week(df$date)

# Time-of-day bucket
df$time_of_day_bucket <- case_when(
  df$hour >= 0  & df$hour < 6  ~ "Night",
  df$hour >= 6  & df$hour < 12 ~ "Morning",
  df$hour >= 12 & df$hour < 18 ~ "Afternoon",
  df$hour >= 18 & df$hour <= 23 ~ "Evening",
  TRUE ~ "Unknown"
)

cat("  Created: hour, day_name, is_weekend, month, week_of_year, time_of_day_bucket\n")

# =============================================================================
# Step 3.3: Compute Trip-Level Delay Features
# =============================================================================
cat("\nStep 3.3: Computing trip-level delay features...\n")

# Ensure delay columns are numeric
df$actual_departure_delay_min <- as.numeric(as.character(df$actual_departure_delay_min))
df$actual_arrival_delay_min   <- as.numeric(as.character(df$actual_arrival_delay_min))

# Total delay
df$total_delay <- df$actual_departure_delay_min + df$actual_arrival_delay_min

# Delay change (did delay grow or shrink en-route?)
df$delay_change <- df$actual_arrival_delay_min - df$actual_departure_delay_min

# Early departure/arrival flags
df$is_early_departure <- ifelse(df$actual_departure_delay_min < 0, 1, 0)
df$is_early_arrival   <- ifelse(df$actual_arrival_delay_min < 0, 1, 0)

# Scheduled trip duration (minutes)
df$trip_duration_scheduled <- as.numeric(difftime(df$sched_arr_datetime,
                                                   df$sched_dep_datetime,
                                                   units = "mins"))

# Handle NA or negative durations (data issues)
df$trip_duration_scheduled[is.na(df$trip_duration_scheduled) |
                             df$trip_duration_scheduled <= 0] <- 
  median(df$trip_duration_scheduled, na.rm = TRUE)

# Severe delay flag
df$severe_delay <- ifelse(df$actual_arrival_delay_min > 15, 1, 0)

cat("  Created: total_delay, delay_change, is_early_departure, is_early_arrival,\n")
cat("           trip_duration_scheduled, severe_delay\n")

# =============================================================================
# Step 3.4: Encode Categorical Variables (Label Encoding)
# =============================================================================
cat("\nStep 3.4: Encoding categorical variables...\n")

# Transport type encoding
transport_map <- c("Bus" = 1, "Metro" = 2, "Train" = 3, "Tram" = 4)
df$transport_type_enc <- transport_map[as.character(df$transport_type)]

# Weather condition encoding (ordinal by severity)
weather_map <- c("Clear" = 1, "Cloudy" = 2, "Fog" = 3,
                 "Rain" = 4, "Snow" = 5, "Storm" = 6)
df$weather_condition_enc <- weather_map[as.character(df$weather_condition)]

# Event type encoding
event_map <- c("None" = 0, "Concert" = 1, "Festival" = 2,
               "Parade" = 3, "Protest" = 4, "Sports" = 5)
df$event_type_enc <- event_map[as.character(df$event_type)]

# Season encoding
season_map <- c("Winter" = 1, "Spring" = 2, "Summer" = 3, "Autumn" = 4)
df$season_enc <- season_map[as.character(df$season)]

cat("  Created: transport_type_enc, weather_condition_enc, event_type_enc, season_enc\n")

# =============================================================================
# Step 3.5: Create Interaction & Derived Weather Features
# =============================================================================
cat("\nStep 3.5: Creating interaction and derived features...\n")

# Weather severity index (weighted combo)
# Higher = more severe weather conditions
storm_indicator <- ifelse(as.character(df$weather_condition) %in% c("Storm", "Snow"), 1, 0)
df$weather_severity_index <- round(
  0.3 * (df$wind_speed_kmh / max(df$wind_speed_kmh, na.rm = TRUE)) +
  0.4 * (df$precipitation_mm / max(df$precipitation_mm + 0.01, na.rm = TRUE)) +
  0.3 * storm_indicator,
  4
)

# Feels-like impact (wind chill proxy)
df$feels_like_impact <- round(df$temperature_C * df$wind_speed_kmh / 100, 4)

# Event impact score
df$event_impact_score <- round(
  df$event_attendance_est * df$traffic_congestion_index / 1000, 4
)

# Congestion × weather combo
df$congestion_weather_combo <- round(
  df$traffic_congestion_index * df$weather_severity_index, 4
)

cat("  Created: weather_severity_index, feels_like_impact, event_impact_score,\n")
cat("           congestion_weather_combo\n")

# =============================================================================
# Step 3.6: Aggregate by Station (Origin)
# =============================================================================
cat("\nStep 3.6: Aggregating by origin station...\n")

station_stats <- df %>%
  group_by(origin_station) %>%
  summarise(
    station_avg_dep_delay = round(mean(actual_departure_delay_min, na.rm = TRUE), 2),
    station_avg_arr_delay = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    station_delay_rate    = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    station_total_trips   = n(),
    station_sd_delay      = round(sd(actual_arrival_delay_min, na.rm = TRUE), 2),
    station_max_delay     = max(actual_arrival_delay_min, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(station_stats, "output/reports/station_origin_statistics.csv")
cat(sprintf("  Saved station statistics for %d origin stations\n", nrow(station_stats)))

# Merge station-level features back to main df
df <- df %>%
  left_join(station_stats %>% select(origin_station, station_avg_dep_delay,
                                      station_delay_rate),
            by = "origin_station")

# =============================================================================
# Step 3.7: Aggregate by Route
# =============================================================================
cat("\nStep 3.7: Aggregating by route...\n")

route_stats <- df %>%
  group_by(route_id) %>%
  summarise(
    route_avg_delay        = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    route_delay_rate       = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    route_total_trips      = n(),
    route_delay_variability = round(sd(actual_arrival_delay_min, na.rm = TRUE), 2),
    .groups = "drop"
  )

write_csv(route_stats, "output/reports/route_statistics.csv")
cat(sprintf("  Saved route statistics for %d routes\n", nrow(route_stats)))

# Merge route-level features back to main df
df <- df %>%
  left_join(route_stats %>% select(route_id, route_avg_delay, route_delay_rate),
            by = "route_id",
            suffix = c("_station", "_route"))

# =============================================================================
# Step 3.8: Aggregate by Hour x Day Type
# =============================================================================
cat("\nStep 3.8: Aggregating by hour × day type...\n")

hourly_stats <- df %>%
  group_by(hour, is_weekend) %>%
  summarise(
    avg_delay_by_hour_day  = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_rate_by_hour_day = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    trip_count_by_hour_day = n(),
    .groups = "drop"
  )

write_csv(hourly_stats, "output/reports/hourly_day_statistics.csv")
cat(sprintf("  Saved hourly × day-type statistics (%d groups)\n", nrow(hourly_stats)))

# =============================================================================
# Step 3.9: Normalize / Standardize Numeric Features
# =============================================================================
cat("\nStep 3.9: Normalizing numeric features...\n")

# Select numeric columns to scale
scale_cols <- c("actual_departure_delay_min", "actual_arrival_delay_min",
                "temperature_C", "humidity_percent", "wind_speed_kmh",
                "precipitation_mm", "event_attendance_est",
                "traffic_congestion_index", "total_delay", "delay_change",
                "trip_duration_scheduled", "weather_severity_index",
                "feels_like_impact", "event_impact_score",
                "congestion_weather_combo")

# Z-score standardization (for PCA & clustering)
scaling_params <- data.frame(
  Column = scale_cols,
  Mean   = sapply(df[scale_cols], function(x) round(mean(as.numeric(x), na.rm = TRUE), 4)),
  SD     = sapply(df[scale_cols], function(x) round(sd(as.numeric(x), na.rm = TRUE), 4)),
  Min    = sapply(df[scale_cols], function(x) round(min(as.numeric(x), na.rm = TRUE), 4)),
  Max    = sapply(df[scale_cols], function(x) round(max(as.numeric(x), na.rm = TRUE), 4)),
  stringsAsFactors = FALSE
)

# Create z-score columns
for (col in scale_cols) {
  col_mean <- mean(as.numeric(df[[col]]), na.rm = TRUE)
  col_sd   <- sd(as.numeric(df[[col]]), na.rm = TRUE)
  if (col_sd > 0) {
    df[[paste0(col, "_z")]] <- round((as.numeric(df[[col]]) - col_mean) / col_sd, 4)
  } else {
    df[[paste0(col, "_z")]] <- 0
  }
}

# Create min-max scaled columns (for LSTM)
for (col in scale_cols) {
  col_min <- min(as.numeric(df[[col]]), na.rm = TRUE)
  col_max <- max(as.numeric(df[[col]]), na.rm = TRUE)
  range_val <- col_max - col_min
  if (range_val > 0) {
    df[[paste0(col, "_mm")]] <- round((as.numeric(df[[col]]) - col_min) / range_val, 4)
  } else {
    df[[paste0(col, "_mm")]] <- 0
  }
}

# Save scaling parameters
write_csv(scaling_params, "output/reports/scaling_parameters.csv")
saveRDS(scaling_params, "output/models/scaling_params.rds")
cat("  Z-score and Min-Max columns created for all numeric features.\n")
cat("  Saved: output/reports/scaling_parameters.csv\n")

# =============================================================================
# Step 3.10: Save Transformed Dataset & Feature Dictionary
# =============================================================================
cat("\nStep 3.10: Saving transformed dataset and feature dictionary...\n")

write_csv(df, "output/cleaned/transport_delays_transformed.csv")
saveRDS(df, "output/cleaned/transport_delays_transformed.rds")
cat(sprintf("  Saved transformed data: %d rows x %d columns\n", nrow(df), ncol(df)))

# Create feature dictionary
feature_dict <- data.frame(
  Column      = names(df),
  Type        = sapply(df, function(x) paste(class(x), collapse = "/")),
  Source      = sapply(names(df), function(col) {
    if (col %in% c("trip_id","date","time","transport_type","route_id",
                    "origin_station","destination_station","scheduled_departure",
                    "scheduled_arrival","actual_departure_delay_min",
                    "actual_arrival_delay_min","weather_condition","temperature_C",
                    "humidity_percent","wind_speed_kmh","precipitation_mm",
                    "event_type","event_attendance_est","traffic_congestion_index",
                    "holiday","peak_hour","weekday","season","delayed"))
      return("Original")
    else if (grepl("_z$", col)) return("Z-score normalization")
    else if (grepl("_mm$", col)) return("Min-Max normalization")
    else if (grepl("_enc$", col)) return("Label encoding")
    else if (col %in% c("station_avg_dep_delay","station_delay_rate"))
      return("Station aggregation")
    else if (col %in% c("route_avg_delay","route_delay_rate_route"))
      return("Route aggregation")
    else return("Feature engineering")
  }),
  stringsAsFactors = FALSE
)

write_csv(feature_dict, "output/reports/feature_dictionary.csv")
cat("  Saved: output/reports/feature_dictionary.csv\n")

cat(sprintf("\n  Transformation summary: %d original → %d total columns\n",
            24, ncol(df)))

cat("\n=== PHASE 3 COMPLETE ===\n")
