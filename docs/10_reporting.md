# Phase 10: Reporting & Power BI Exports (`10_reporting.R`)

## Purpose
The **final delivery phase**. Prepares pre-aggregated, Power BI-ready CSVs, generates a combined R dashboard, produces a final summary of statistics, and catalogs every output file generated across all phases.

---

## Step-by-Step Breakdown

### Step 10.1: Power BI Data Exports
Seven targeted tables are created — each pre-aggregated to the exact grain that each Power BI visual needs.

**Why pre-aggregate instead of using raw data in Power BI?**
Power BI DAX (its query language) can aggregate on the fly, but pre-computed aggregations:
1. Load and refresh significantly faster
2. Reduce the chance of DAX errors
3. Allow consistent definitions across the team

---

**Table 1: Overall KPI Summary**
```r
kpi_summary <- data.frame(
  Metric = c("Total Trips", "Delayed Trips", "On-Time Trips",
             "Delay Rate (%)", "Avg Arrival Delay (min)",
             "Avg Departure Delay (min)", "Max Arrival Delay (min)",
             "Unique Stations", "Unique Routes", "Date Range Start", "Date Range End"),
  Value = c(
    nrow(df),                                                   # e.g., 9975
    sum(df$delayed_num, na.rm = TRUE),                          # e.g., 3471
    sum(df$delayed_num == 0, na.rm = TRUE),                     # e.g., 6504
    round(mean(df$delayed_num, na.rm = TRUE) * 100, 2),         # e.g., 34.8
    round(mean(df$actual_arrival_delay_min, na.rm = TRUE), 2),  # e.g., 6.23
    round(mean(df$actual_departure_delay_min, na.rm = TRUE), 2),# e.g., 5.11
    max(df$actual_arrival_delay_min, na.rm = TRUE),             # e.g., 27 (winsorized)
    n_distinct(c(df$origin_station, df$destination_station)),   # e.g., 85
    n_distinct(df$route_id),                                    # e.g., 45
    as.character(min(df$date, na.rm = TRUE)),                   # "2024-01-01"
    as.character(max(df$date, na.rm = TRUE))                    # "2024-12-31"
  )
)
```

**Example output (first 5 rows):**
```
                    Metric   Value
               Total Trips    9975
             Delayed Trips    3471
            On-Time Trips     6504
           Delay Rate (%)   34.80
  Avg Arrival Delay (min)    6.23
```

---

**Table 2: Station Rankings**
```r
station_rankings <- df %>%
  group_by(origin_station) %>%
  summarise(
    total_trips = n(),
    avg_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_rate  = round(mean(delayed_num, na.rm = TRUE) * 100, 2)
  ) %>%
  arrange(desc(avg_delay)) %>%
  mutate(rank = row_number())   # Rank 1 = worst station
```

**Usage in Power BI:** Connect this to a Map visual using `origin_station` as location → size/colour by `avg_delay` → instantly see geographic bottleneck heatmap.

---

**Table 3: Hourly Patterns**
```r
hourly_patterns <- df %>%
  group_by(hour, day_type = ifelse(is_weekend == 1, "Weekend", "Weekday")) %>%
  summarise(
    avg_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    delay_count = sum(delayed_num, na.rm = TRUE),
    total_trips = n(),
    delay_rate  = round(mean(delayed_num, na.rm = TRUE) * 100, 2)
  )
```

**Example output:**
```
hour  day_type  avg_delay  delay_count  total_trips  delay_rate
   0   Weekday       2.10           24          142       16.9%
   7   Weekday       9.80          198          342       57.9%  ← Morning rush
   8   Weekday      12.40          256          381       67.2%  ← Worst hour, weekdays
   9   Weekend       4.20           45          198       22.7%  ← Weekends much better
```

**Usage in Power BI:** Line chart with `hour` on X-axis, `avg_delay` on Y-axis, filtered by `day_type` — directly shows the rush-hour pattern.

---

### Step 10.2: Combined R Dashboard (2×2 Grid)
Creates a single, presentation-ready PNG combining the four most important visualizations.

```r
# Top-left: Delay distribution
p1 <- ggplot(df, aes(x = actual_arrival_delay_min)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "white", alpha = 0.8) +
  labs(title = "Delay Distribution", x = "Arrival Delay (min)")

# Top-right: Peak vs Off-Peak average delay
peak_df <- df %>%
  mutate(peak_label = ifelse(as.character(peak_hour) == "1", "Peak", "Off-Peak")) %>%
  group_by(peak_label) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE))

p2 <- ggplot(peak_df, aes(x = peak_label, y = avg_delay, fill = peak_label)) +
  geom_col(alpha = 0.8, width = 0.5) +
  geom_text(aes(label = round(avg_delay, 1)), vjust = -0.5)  # Show values on bars

# Bottom-left: Top 10 delayed stations (yellow → red gradient)
p3 <- ggplot(top_stations, aes(x = reorder(origin_station, avg_delay),
                                y = avg_delay, fill = avg_delay)) +
  geom_col() + coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red") # Hot stations in red

# Bottom-right: Hourly pattern line chart
p4 <- ggplot(hourly_line, aes(x = hour, y = avg_delay)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  scale_x_continuous(breaks = seq(0, 23, 2))

# Combine into a 2×2 dashboard
dashboard <- grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2,
                           top = "Public Transport Delay Analysis — Dashboard")
ggsave("output/plots/comprehensive_delay_dashboard.png", dashboard,
       width = 16, height = 12, dpi = 300)
```

**What this dashboard communicates at a glance:**
- Top-left: Shape of delays (is the distribution right-skewed?)
- Top-right: How much peak hours worsen delays
- Bottom-left: Which stations are the worst offenders
- Bottom-right: Delay pattern throughout the day

---

### Step 10.3: Output Manifest
Automatically catalogs every file generated by all phases.

```r
all_outputs <- list.files("output", recursive = TRUE, full.names = FALSE)

output_manifest <- data.frame(
  Category = sapply(all_outputs, function(f) {
    if (grepl("^cleaned/", f)) "Cleaned Data"
    else if (grepl("^plots/", f)) "Plots"
    else if (grepl("^models/", f)) "Models"
    else if (grepl("^reports/", f)) "Reports"
    else "Other"
  }),
  File = all_outputs
)
```

**Example manifest excerpt:**
```
      Category                                  File
  Cleaned Data   cleaned/transport_delays_transformed.rds
  Cleaned Data   cleaned/pca_transformed_data.csv
        Plots   plots/hour_weekday_heatmap.png
        Plots   plots/roc_curves_comparison.png
       Models   models/best_classification_model.rds
       Models   models/lstm_model.h5
      Reports   reports/model_comparison.csv
      Reports   reports/station_network_metrics.csv

Total output files: 47
  Cleaned data: 8 files
  Plots: 24 files
  Models: 9 files
  Reports: 6 files
```

---

## Outputs
| File | Description |
|---|---|
| `output/reports/powerbi_overall_summary.csv` | KPI cards table |
| `output/reports/powerbi_station_rankings.csv` | Station leaderboard |
| `output/reports/powerbi_hourly_patterns.csv` | Hourly delay patterns table |
| `output/reports/powerbi_weather_impact.csv` | Weather breakdown table |
| `output/reports/powerbi_transformed_full.csv` | Trimmed full dataset for slicers |
| `output/plots/comprehensive_delay_dashboard.png` | 2×2 combined dashboard |
| `output/reports/output_manifest.csv` | Complete catalog of all pipeline outputs |

---

## 💡 Presentation Talking Points
> "We did not export the full 60-column dataset to Power BI — we exported 7 purpose-built tables, each aggregated to exactly the grain each visual needs. This makes the Power BI dashboard fast, clean, and maintainable."

> "The output manifest at the end of the pipeline answers the question 'what did this project produce?' with a single file. 47 files across 4 categories, all generated automatically from one `source()` call."
