# =============================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# Script: 04_eda.R
# Purpose: Generate comprehensive visualizations covering delay distributions,
#          temporal patterns, weather/event impact, station/route analysis,
#          correlation analysis, and target variable analysis.
# =============================================================================

cat("=== PHASE 4: EXPLORATORY DATA ANALYSIS ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(pheatmap)
library(gridExtra)
library(scales)
library(lubridate)

# --- Load transformed data ----------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
cat(sprintf("Loaded transformed data: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# Ensure delayed is treated as numeric for calculations then as factor for plots
df$delayed_num <- as.numeric(as.character(df$delayed))

# Helper: save a ggplot
save_plot <- function(p, filename, w = 10, h = 7) {
  ggsave(filename = file.path("output/plots", filename),
         plot = p, width = w, height = h, dpi = 300)
  cat(sprintf("  Saved: output/plots/%s\n", filename))
}

# =============================================================================
# Step 4.1: Delay Distribution Analysis
# =============================================================================
cat("Step 4.1: Delay distribution analysis...\n")

# Histogram of arrival delay
p1 <- ggplot(df, aes(x = actual_arrival_delay_min)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(aes(xintercept = mean(actual_arrival_delay_min, na.rm = TRUE)),
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(aes(xintercept = median(actual_arrival_delay_min, na.rm = TRUE)),
             color = "orange", linetype = "dashed", linewidth = 1) +
  labs(title = "Distribution of Arrival Delay (minutes)",
       subtitle = "Red = Mean | Orange = Median",
       x = "Arrival Delay (min)", y = "Frequency") +
  theme_minimal(base_size = 13)
save_plot(p1, "delay_distribution_histogram.png")

# Histogram of departure delay
p2 <- ggplot(df, aes(x = actual_departure_delay_min)) +
  geom_histogram(bins = 50, fill = "coral", color = "white", alpha = 0.8) +
  geom_vline(aes(xintercept = mean(actual_departure_delay_min, na.rm = TRUE)),
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "Distribution of Departure Delay (minutes)",
       x = "Departure Delay (min)", y = "Frequency") +
  theme_minimal(base_size = 13)
save_plot(p2, "departure_delay_distribution.png")

# Box plot: arrival delay by transport type
p3 <- ggplot(df, aes(x = transport_type, y = actual_arrival_delay_min,
                      fill = transport_type)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  labs(title = "Arrival Delay by Transport Type",
       x = "Transport Type", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")
save_plot(p3, "delay_by_transport_type.png")

# Box plot: arrival delay by weekend/weekday
df$day_type_label <- ifelse(df$is_weekend == 1, "Weekend", "Weekday")
p4 <- ggplot(df, aes(x = day_type_label, y = actual_arrival_delay_min,
                      fill = day_type_label)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Arrival Delay: Weekday vs Weekend",
       x = "", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("Weekday" = "steelblue", "Weekend" = "coral"))
save_plot(p4, "delay_weekday_vs_weekend.png")

# =============================================================================
# Step 4.2: Temporal Pattern Analysis
# =============================================================================
cat("\nStep 4.2: Temporal pattern analysis...\n")

# Average delay by hour of day, faceted by day type
hourly_avg <- df %>%
  group_by(hour, day_type_label) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE),
            .groups = "drop")

p5 <- ggplot(hourly_avg, aes(x = hour, y = avg_delay, color = day_type_label)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  labs(title = "Average Arrival Delay by Hour of Day",
       x = "Hour of Day", y = "Avg Arrival Delay (min)", color = "Day Type") +
  scale_x_continuous(breaks = 0:23) +
  theme_minimal(base_size = 13) +
  scale_color_manual(values = c("Weekday" = "steelblue", "Weekend" = "coral"))
save_plot(p5, "hourly_delay_pattern.png")

# Delay rate by day of week
daily_rate <- df %>%
  mutate(day_name = factor(day_name, levels = c("Monday","Tuesday","Wednesday",
                                                  "Thursday","Friday","Saturday","Sunday"))) %>%
  group_by(day_name) %>%
  summarise(delay_rate = mean(delayed_num, na.rm = TRUE) * 100, .groups = "drop")

p6 <- ggplot(daily_rate, aes(x = day_name, y = delay_rate, fill = day_name)) +
  geom_col(alpha = 0.8) +
  labs(title = "Delay Rate by Day of Week",
       x = "", y = "Delay Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set3")
save_plot(p6, "delay_rate_by_day.png")

# Heatmap: hour × weekday delay intensity
heat_data <- df %>%
  mutate(day_name = factor(day_name, levels = c("Monday","Tuesday","Wednesday",
                                                  "Thursday","Friday","Saturday","Sunday"))) %>%
  group_by(day_name, hour) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = hour, values_from = avg_delay, values_fill = 0)

heat_matrix <- as.matrix(heat_data[, -1])
rownames(heat_matrix) <- heat_data$day_name

png("output/plots/hour_weekday_heatmap.png", width = 1400, height = 600, res = 150)
pheatmap(heat_matrix,
         cluster_rows = FALSE, cluster_cols = FALSE,
         color = colorRampPalette(c("white", "yellow", "orange", "red"))(50),
         main = "Average Arrival Delay: Hour × Day of Week",
         fontsize = 10)
dev.off()
cat("  Saved: output/plots/hour_weekday_heatmap.png\n")

# Delay trend across dates
daily_trend <- df %>%
  group_by(date) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop")

p7 <- ggplot(daily_trend, aes(x = date, y = avg_delay)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 0.8) +
  labs(title = "Daily Average Arrival Delay Trend",
       x = "Date", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13)
save_plot(p7, "daily_delay_trend.png")

# =============================================================================
# Step 4.3: Weather Impact Analysis
# =============================================================================
cat("\nStep 4.3: Weather impact analysis...\n")

# Average delay by weather condition
weather_avg <- df %>%
  group_by(weather_condition) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE),
            delay_rate = mean(delayed_num, na.rm = TRUE) * 100,
            .groups = "drop") %>%
  arrange(desc(avg_delay))

p8 <- ggplot(weather_avg, aes(x = reorder(weather_condition, avg_delay),
                               y = avg_delay, fill = weather_condition)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  labs(title = "Average Delay by Weather Condition",
       x = "", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")
save_plot(p8, "weather_impact_bar.png")

# Scatter: temperature vs delay
p9 <- ggplot(df, aes(x = temperature_C, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  labs(title = "Temperature vs Arrival Delay",
       x = "Temperature (°C)", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13)
save_plot(p9, "temperature_vs_delay.png")

# Scatter: precipitation vs delay
p10 <- ggplot(df, aes(x = precipitation_mm, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "darkgreen") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  labs(title = "Precipitation vs Arrival Delay",
       x = "Precipitation (mm)", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13)
save_plot(p10, "precipitation_vs_delay.png")

# Scatter: wind speed vs delay
p11 <- ggplot(df, aes(x = wind_speed_kmh, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "purple") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  labs(title = "Wind Speed vs Arrival Delay",
       x = "Wind Speed (km/h)", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13)
save_plot(p11, "wind_speed_vs_delay.png")

# =============================================================================
# Step 4.4: Event & Traffic Impact Analysis
# =============================================================================
cat("\nStep 4.4: Event & traffic impact analysis...\n")

# Average delay by event type
event_avg <- df %>%
  group_by(event_type) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(avg_delay))

p12 <- ggplot(event_avg, aes(x = reorder(event_type, avg_delay),
                              y = avg_delay, fill = event_type)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  labs(title = "Average Delay by Event Type",
       x = "", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set1")
save_plot(p12, "event_impact_bar.png")

# Scatter: traffic congestion vs delay
p13 <- ggplot(df, aes(x = traffic_congestion_index, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "darkorange") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Traffic Congestion Index vs Arrival Delay",
       x = "Traffic Congestion Index", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13)
save_plot(p13, "congestion_vs_delay.png")

# Box plot: holiday effect
p14 <- ggplot(df, aes(x = factor(holiday, labels = c("No Holiday", "Holiday")),
                       y = actual_arrival_delay_min,
                       fill = factor(holiday))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Arrival Delay: Holiday vs Non-Holiday",
       x = "", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"))
save_plot(p14, "holiday_effect.png")

# Box plot: peak hour effect
p15 <- ggplot(df, aes(x = factor(peak_hour, labels = c("Off-Peak", "Peak Hour")),
                       y = actual_arrival_delay_min,
                       fill = factor(peak_hour))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Arrival Delay: Peak vs Off-Peak Hours",
       x = "", y = "Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("0" = "lightgreen", "1" = "tomato"))
save_plot(p15, "peak_hour_effect.png")

# =============================================================================
# Step 4.5: Station & Route Analysis
# =============================================================================
cat("\nStep 4.5: Station & route analysis...\n")

# Top 15 origin stations by avg delay
station_origin_avg <- df %>%
  group_by(origin_station) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  top_n(15, avg_delay) %>%
  arrange(desc(avg_delay))

p16 <- ggplot(station_origin_avg, aes(x = reorder(origin_station, avg_delay),
                                       y = avg_delay, fill = avg_delay)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red") +
  labs(title = "Top 15 Origin Stations by Average Delay",
       x = "", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
save_plot(p16, "top15_origin_stations.png")

# Top 15 destination stations by avg delay
station_dest_avg <- df %>%
  group_by(destination_station) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  top_n(15, avg_delay) %>%
  arrange(desc(avg_delay))

p17 <- ggplot(station_dest_avg, aes(x = reorder(destination_station, avg_delay),
                                     y = avg_delay, fill = avg_delay)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Top 15 Destination Stations by Average Delay",
       x = "", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
save_plot(p17, "top15_destination_stations.png")

# Top 10 routes by avg delay
route_avg <- df %>%
  group_by(route_id) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  top_n(10, avg_delay) %>%
  arrange(desc(avg_delay))

p18 <- ggplot(route_avg, aes(x = reorder(route_id, avg_delay),
                              y = avg_delay, fill = avg_delay)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(title = "Top 10 Routes by Average Delay",
       x = "", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
save_plot(p18, "top10_routes_delay.png")

# Station-hour heatmap (top 20 origin stations)
top20_stations <- df %>%
  group_by(origin_station) %>%
  summarise(n = n(), .groups = "drop") %>%
  top_n(20, n) %>%
  pull(origin_station)

station_hour_data <- df %>%
  filter(origin_station %in% top20_stations) %>%
  group_by(origin_station, hour) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = hour, values_from = avg_delay, values_fill = 0)

station_hour_matrix <- as.matrix(station_hour_data[, -1])
rownames(station_hour_matrix) <- station_hour_data$origin_station

png("output/plots/station_hour_heatmap.png", width = 1600, height = 800, res = 150)
pheatmap(station_hour_matrix,
         cluster_rows = TRUE, cluster_cols = FALSE,
         color = colorRampPalette(c("white", "yellow", "orange", "red"))(50),
         main = "Station × Hour Heatmap: Average Arrival Delay",
         fontsize = 9)
dev.off()
cat("  Saved: output/plots/station_hour_heatmap.png\n")

# =============================================================================
# Step 4.6: Correlation Analysis
# =============================================================================
cat("\nStep 4.6: Correlation analysis...\n")

# Select numeric columns for correlation
cor_cols <- c("actual_departure_delay_min", "actual_arrival_delay_min",
              "temperature_C", "humidity_percent", "wind_speed_kmh",
              "precipitation_mm", "event_attendance_est",
              "traffic_congestion_index", "weekday", "total_delay",
              "delay_change", "trip_duration_scheduled",
              "weather_severity_index", "event_impact_score")

# Filter to only columns that exist
cor_cols <- cor_cols[cor_cols %in% names(df)]
cor_data <- df[, cor_cols]
cor_data <- cor_data %>% mutate(across(everything(), as.numeric))

cor_matrix <- cor(cor_data, use = "complete.obs")

png("output/plots/correlation_matrix.png", width = 1200, height = 1000, res = 150)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.7, tl.col = "black",
         addCoef.col = "black", number.cex = 0.55,
         title = "Correlation Matrix — Numeric Features",
         mar = c(0, 0, 2, 0))
dev.off()
cat("  Saved: output/plots/correlation_matrix.png\n")

# Report key correlations
cat("\n  Key correlations with actual_arrival_delay_min:\n")
arr_cors <- cor_matrix[, "actual_arrival_delay_min"]
arr_cors <- sort(arr_cors, decreasing = TRUE)
for (i in seq_along(arr_cors)) {
  cat(sprintf("    %s: %.3f\n", names(arr_cors)[i], arr_cors[i]))
}

# =============================================================================
# Step 4.7: Target Variable Analysis
# =============================================================================
cat("\nStep 4.7: Target variable analysis...\n")

# Class distribution
class_counts <- df %>% count(delayed)
class_pct <- class_counts %>% mutate(pct = round(n / sum(n) * 100, 1))
cat("  Class distribution:\n")
print(class_pct)

p19 <- ggplot(class_pct, aes(x = delayed, y = n, fill = delayed)) +
  geom_col(alpha = 0.8, width = 0.5) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), vjust = -0.5, size = 5) +
  labs(title = "Target Variable Distribution (Delayed)",
       x = "Delayed", y = "Count") +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels = c("On-Time", "Delayed")) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
save_plot(p19, "target_class_distribution.png")

# Cross-tabulations
cat("\n  Cross-tabulation: delayed vs transport_type\n")
print(table(df$delayed, df$transport_type))

cat("\n  Cross-tabulation: delayed vs weather_condition\n")
print(table(df$delayed, df$weather_condition))

cat("\n  Cross-tabulation: delayed vs season\n")
print(table(df$delayed, df$season))

cat("\n  Cross-tabulation: delayed vs peak_hour\n")
print(table(df$delayed, df$peak_hour))

# =============================================================================
# Step 4.8: Summary & Save
# =============================================================================
cat("\nStep 4.8: All EDA plots saved.\n")

# List all saved plots
plot_files <- list.files("output/plots", pattern = "\\.png$", full.names = FALSE)
cat(sprintf("  Total plots generated: %d\n", length(plot_files)))
for (f in plot_files) cat(sprintf("    - %s\n", f))

cat("\n=== PHASE 4 COMPLETE ===\n")
