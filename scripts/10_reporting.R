# =============================================================================
# PHASE 10: VISUALIZATION, POWER BI DATA EXPORTS & REPORTING
# Script: 10_reporting.R
# Purpose: Prepare Power BI data exports, create a combined R dashboard,
#          and generate a final project summary.
# =============================================================================

cat("=== PHASE 10: VISUALIZATION, POWER BI EXPORTS & REPORTING ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)

# --- Load data ----------------------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
df$delayed_num <- as.numeric(as.character(df$delayed))

cat(sprintf("Loaded: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 10.1: Prepare Power BI Data Exports
# =============================================================================
cat("Step 10.1: Preparing Power BI data exports...\n")

# --- 1. Overall Summary (KPIs) ---
kpi_summary <- data.frame(
  Metric = c("Total Trips", "Delayed Trips", "On-Time Trips",
             "Delay Rate (%)", "Avg Arrival Delay (min)",
             "Avg Departure Delay (min)", "Max Arrival Delay (min)",
             "Unique Stations", "Unique Routes", "Date Range Start",
             "Date Range End"),
  Value = c(
    nrow(df),
    sum(df$delayed_num, na.rm = TRUE),
    sum(df$delayed_num == 0, na.rm = TRUE),
    round(mean(df$delayed_num, na.rm = TRUE) * 100, 2),
    round(mean(df$actual_arrival_delay_min, na.rm = TRUE), 2),
    round(mean(df$actual_departure_delay_min, na.rm = TRUE), 2),
    max(df$actual_arrival_delay_min, na.rm = TRUE),
    n_distinct(c(df$origin_station, df$destination_station)),
    n_distinct(df$route_id),
    as.character(min(df$date, na.rm = TRUE)),
    as.character(max(df$date, na.rm = TRUE))
  ),
  stringsAsFactors = FALSE
)
write_csv(kpi_summary, "output/reports/powerbi_overall_summary.csv")
cat("  Saved: powerbi_overall_summary.csv\n")

# --- 2. Station Rankings ---
station_rankings <- df %>%
  group_by(origin_station) %>%
  summarise(
    total_trips = n(),
    avg_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_rate  = round(mean(delayed_num, na.rm = TRUE) * 100, 2),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_delay)) %>%
  mutate(rank = row_number())
write_csv(station_rankings, "output/reports/powerbi_station_rankings.csv")
cat("  Saved: powerbi_station_rankings.csv\n")

# --- 3. Hourly Patterns ---
hourly_patterns <- df %>%
  group_by(hour, day_type = ifelse(is_weekend == 1, "Weekend", "Weekday")) %>%
  summarise(
    avg_delay    = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_count  = sum(delayed_num, na.rm = TRUE),
    total_trips  = n(),
    delay_rate   = round(mean(delayed_num, na.rm = TRUE) * 100, 2),
    .groups = "drop"
  )
write_csv(hourly_patterns, "output/reports/powerbi_hourly_patterns.csv")
cat("  Saved: powerbi_hourly_patterns.csv\n")

# --- 4. Cluster Summary ---
tryCatch({
  cluster_profiles <- read_csv("output/reports/cluster_profiles.csv", show_col_types = FALSE)
  write_csv(cluster_profiles, "output/reports/powerbi_cluster_summary.csv")
  cat("  Saved: powerbi_cluster_summary.csv\n")
}, error = function(e) {
  cat("  Cluster profiles not available yet — skipping.\n")
})

# --- 5. Weather Impact ---
weather_impact <- df %>%
  group_by(weather_condition) %>%
  summarise(
    avg_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_rate  = round(mean(delayed_num, na.rm = TRUE) * 100, 2),
    total_trips = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_delay))
write_csv(weather_impact, "output/reports/powerbi_weather_impact.csv")
cat("  Saved: powerbi_weather_impact.csv\n")

# --- 6. Model Comparison ---
tryCatch({
  model_comp <- read_csv("output/reports/model_comparison.csv", show_col_types = FALSE)
  write_csv(model_comp, "output/reports/powerbi_model_comparison.csv")
  cat("  Saved: powerbi_model_comparison.csv\n")
}, error = function(e) {
  cat("  Model comparison not available yet — skipping.\n")
})

# --- 7. Full Transformed Dataset for Power BI Slicing ---
# Select key columns only (not all 60+ columns)
powerbi_full <- df %>%
  select(trip_id, date, hour, transport_type, route_id,
         origin_station, destination_station,
         actual_departure_delay_min, actual_arrival_delay_min,
         weather_condition, temperature_C, precipitation_mm,
         event_type, event_attendance_est, traffic_congestion_index,
         holiday, peak_hour, weekday, season, delayed,
         total_delay, delay_change, time_of_day_bucket, is_weekend,
         weather_severity_index, event_impact_score)
write_csv(powerbi_full, "output/reports/powerbi_transformed_full.csv")
cat("  Saved: powerbi_transformed_full.csv\n")

# =============================================================================
# Step 10.2: Create R Combined Dashboard Plot (2x2)
# =============================================================================
cat("\nStep 10.2: Creating combined R dashboard...\n")

# Top-left: Delay distribution histogram
p1 <- ggplot(df, aes(x = actual_arrival_delay_min)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "white", alpha = 0.8) +
  labs(title = "Delay Distribution", x = "Arrival Delay (min)", y = "Count") +
  theme_minimal(base_size = 11)

# Top-right: Peak vs Off-Peak average delay
peak_df <- df %>%
  mutate(peak_label = ifelse(as.character(peak_hour) == "1", "Peak", "Off-Peak")) %>%
  group_by(peak_label) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop")

p2 <- ggplot(peak_df, aes(x = peak_label, y = avg_delay, fill = peak_label)) +
  geom_col(alpha = 0.8, width = 0.5) +
  geom_text(aes(label = round(avg_delay, 1)), vjust = -0.5) +
  labs(title = "Peak vs Off-Peak Delay", x = "", y = "Avg Delay (min)") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("Off-Peak" = "lightgreen", "Peak" = "tomato"))

# Bottom-left: Top 10 delayed origin stations
top_stations <- df %>%
  group_by(origin_station) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  top_n(10, avg_delay) %>%
  arrange(desc(avg_delay))

p3 <- ggplot(top_stations, aes(x = reorder(origin_station, avg_delay),
                                y = avg_delay, fill = avg_delay)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red") +
  labs(title = "Top 10 Delayed Stations", x = "", y = "Avg Delay (min)") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

# Bottom-right: Hourly delay pattern
hourly_line <- df %>%
  group_by(hour) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop")

p4 <- ggplot(hourly_line, aes(x = hour, y = avg_delay)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  labs(title = "Hourly Delay Pattern", x = "Hour", y = "Avg Delay (min)") +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  theme_minimal(base_size = 11)

# Combine into 2x2 dashboard
dashboard <- grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2,
                           top = "Public Transport Delay Analysis — Dashboard")

ggsave("output/plots/comprehensive_delay_dashboard.png", dashboard,
       width = 16, height = 12, dpi = 300)
cat("  Saved: output/plots/comprehensive_delay_dashboard.png\n")

# =============================================================================
# Step 10.3: Generate Final Project Summary
# =============================================================================
cat("\nStep 10.3: Generating project summary...\n")

# Collect all output files
all_outputs <- list.files("output", recursive = TRUE, full.names = FALSE)

# Categorize
output_manifest <- data.frame(
  Category = sapply(all_outputs, function(f) {
    if (grepl("^cleaned/", f)) "Cleaned Data"
    else if (grepl("^plots/", f)) "Plots"
    else if (grepl("^models/", f)) "Models"
    else if (grepl("^reports/", f)) "Reports"
    else "Other"
  }),
  File = all_outputs,
  stringsAsFactors = FALSE
)
rownames(output_manifest) <- NULL

write_csv(output_manifest, "output/reports/output_manifest.csv")
cat("  Saved: output/reports/output_manifest.csv\n")

cat("\n  Output file manifest:\n")
for (cat_name in unique(output_manifest$Category)) {
  cat(sprintf("\n  [%s]\n", cat_name))
  files <- output_manifest$File[output_manifest$Category == cat_name]
  for (f in files) cat(sprintf("    - %s\n", f))
}

cat(sprintf("\n  Total output files: %d\n", nrow(output_manifest)))

# =============================================================================
# Summary Statistics for Report
# =============================================================================
cat("\n  === PROJECT SUMMARY STATISTICS ===\n")
cat(sprintf("  Total records: %d\n", nrow(df)))
cat(sprintf("  Features (after engineering): %d\n", ncol(df)))
cat(sprintf("  Date range: %s to %s\n",
            min(df$date, na.rm = TRUE), max(df$date, na.rm = TRUE)))
cat(sprintf("  Delay rate: %.1f%%\n", mean(df$delayed_num, na.rm = TRUE) * 100))
cat(sprintf("  Avg arrival delay: %.2f min\n",
            mean(df$actual_arrival_delay_min, na.rm = TRUE)))
cat(sprintf("  Transport types: %s\n",
            paste(levels(df$transport_type), collapse = ", ")))

cat("\n=== PHASE 10 COMPLETE ===\n")
cat("\n====================================\n")
cat("  ALL PHASES COMPLETE!\n")
cat("  Review outputs in the output/ folder.\n")
cat("====================================\n")
