# Phase 4: Exploratory Data Analysis (`04_eda.R`)

## Purpose
Generates **19+ publication-quality visualizations** to reveal delay patterns across time, weather, events, stations, and routes. All plots are saved at 300 DPI. The goal is to build *visual intuition* that validates (or challenges) our assumptions before modelling.

---

## Reusable Plot Helper
To avoid repeating the same `ggsave()` call with the same settings on every plot, a helper function centralises all output settings:

```r
save_plot <- function(p, filename, w = 10, h = 7) {
  ggsave(filename = file.path("output/plots", filename),
         plot = p, width = w, height = h, dpi = 300)
  cat(sprintf("  Saved: output/plots/%s\n", filename))
}
# Usage: save_plot(p1, "delay_distribution_histogram.png")
```

**Why this matters:** If we later need to change all plots to 600 DPI or a different size, we change one line instead of 19+.

---

## Step-by-Step Breakdown

### Step 4.1: Delay Distribution Analysis
Understanding the *shape* of delay distributions reveals whether delays are symmetric (random noise) or skewed (systemic issues).

```r
p1 <- ggplot(df, aes(x = actual_arrival_delay_min)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
  # Add mean (red) and median (orange) reference lines
  geom_vline(aes(xintercept = mean(actual_arrival_delay_min, na.rm = TRUE)),
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(aes(xintercept = median(actual_arrival_delay_min, na.rm = TRUE)),
             color = "orange", linetype = "dashed", linewidth = 1) +
  labs(title = "Distribution of Arrival Delay (minutes)",
       subtitle = "Red = Mean | Orange = Median")
```

**What to look for:**
- If `mean > median` → right-skewed → most trips are on-time, but a few severe outliers pull the average up. This is typical for transit data.
- If `mean ≈ median` → symmetric distribution → rare for real-world transit

**Box plots by transport type:**
```r
# Reveals which transport mode is most delay-prone
p3 <- ggplot(df, aes(x = transport_type, y = actual_arrival_delay_min, fill = transport_type)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  scale_fill_brewer(palette = "Set2")
```

**Example insight:** If Bus has a much longer box (wide IQR) than Metro, it means bus delays are more variable and unpredictable, while metro delays are more consistent.

---

### Step 4.2: Temporal Pattern Analysis
Reveals when delays happen — arguably the most operationally useful insight for transport planners.

**Hour × Weekday Heatmap:**
```r
# 1. Aggregate: average delay per day-of-week × hour combination
heat_data <- df %>%
  group_by(day_name, hour) %>%
  summarise(avg_delay = mean(actual_arrival_delay_min, na.rm = TRUE)) %>%
  pivot_wider(names_from = hour, values_from = avg_delay, values_fill = 0)

heat_matrix <- as.matrix(heat_data[, -1])
rownames(heat_matrix) <- heat_data$day_name

# 2. Render as colour-coded heatmap
pheatmap(heat_matrix,
  cluster_rows = FALSE, cluster_cols = FALSE,  # Preserve time order
  color = colorRampPalette(c("white", "yellow", "orange", "red"))(50),
  main = "Average Arrival Delay: Hour × Day of Week")
```

**Reading the heatmap:**
- ⬜ White cells = little or no delay (e.g., 3 AM Sunday)
- 🟥 Red cells = high average delay (e.g., 8 AM Monday, 5 PM Friday)
- This instantly reveals "rush hour bottleneck windows" as a 7×24 matrix

**Daily trend with LOESS smoothing:**
```r
p7 <- ggplot(daily_trend, aes(x = date, y = avg_delay)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE, color = "red")
  # LOESS = locally weighted smoothing: reveals seasonal/long-term trend
  # The shaded band = 95% confidence interval
```

---

### Step 4.3: Weather Impact Analysis
Quantifies how much each weather type worsens delays.

```r
# Aggregate: average delay and delay rate per weather condition
weather_avg <- df %>%
  group_by(weather_condition) %>%
  summarise(
    avg_delay  = mean(actual_arrival_delay_min, na.rm = TRUE),
    delay_rate = mean(delayed_num, na.rm = TRUE) * 100
  ) %>%
  arrange(desc(avg_delay))  # Sort worst-to-best

# Horizontal bar chart (flipped) — easier to read long category names
p8 <- ggplot(weather_avg, aes(x = reorder(weather_condition, avg_delay),
                                y = avg_delay, fill = weather_condition)) +
  geom_col(alpha = 0.8) +
  coord_flip()  # Flip axes: categories on Y, delay on X
```

**Expected result order (worst to best):**
`Storm > Snow > Rain > Fog > Cloudy > Clear`

**Scatter plot — continuous weather vs delay:**
```r
# Does more precipitation linearly increase delay?
p10 <- ggplot(df, aes(x = precipitation_mm, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "darkgreen") +   # alpha=0.3 handles overplotting
  geom_smooth(method = "loess", se = TRUE, color = "red")
# If the LOESS curve trends upward right, precipitation strongly predicts delay
```

---

### Step 4.4: Event & Traffic Impact
```r
# Traffic congestion: linear regression line shows direct relationship
p13 <- ggplot(df, aes(x = traffic_congestion_index, y = actual_arrival_delay_min)) +
  geom_point(alpha = 0.3, color = "darkorange") +
  geom_smooth(method = "lm", se = TRUE, color = "red")
  # Using lm (linear model) here instead of LOESS assumes a proportional relationship

# Peak hour effect: simple comparison
p15 <- ggplot(df, aes(x = factor(peak_hour, labels = c("Off-Peak", "Peak Hour")),
                       y = actual_arrival_delay_min, fill = factor(peak_hour))) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("0" = "lightgreen", "1" = "tomato"))
```

**Example insight:** If the Peak Hour box is significantly higher than Off-Peak, it validates our `peak_hour` feature as a strong predictor — worth including in the classification model.

---

### Step 4.5: Station & Route Analysis — Finding Bottlenecks Visually
```r
# Colour gradient: yellow (low delay) → red (high delay) — instantly highlights hot stations
p16 <- ggplot(station_origin_avg, aes(x = reorder(origin_station, avg_delay),
                                       y = avg_delay, fill = avg_delay)) +
  geom_col() + coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red")

# Station × Hour heatmap: reveals time-of-day patterns per station
station_hour_matrix <-  # (pivot and aggregate as above)
pheatmap(station_hour_matrix,
  cluster_rows = TRUE,  # Group similar stations together
  cluster_cols = FALSE, # Keep hours in order
  color = colorRampPalette(c("white", "yellow", "orange", "red"))(50))
```

**Reading the station heatmap:**
Stations clustered together at the top have similar delay patterns across the day → they likely share a network segment or serve the same corridor.

---

### Step 4.6: Correlation Matrix
The backbone of feature selection — tells us which raw features are correlated with the delay target.

```r
cor_matrix <- cor(cor_data, use = "complete.obs")   # Pearson correlation

corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black",   # Print correlation coefficients in cells
         number.cex = 0.55,       # Small text to fit all numbers
         tl.cex = 0.7,            # Smaller axis labels
         title = "Correlation Matrix — Numeric Features")
```

**Key values to explain:**
- `correlation = 1.00` → perfect positive correlation (only itself)
- `correlation ≈ 0.70` → strong relationship (feature is very predictive)
- `correlation ≈ 0.00` → no linear relationship (feature may still be useful non-linearly)
- `correlation ≈ -0.50` → moderate negative relationship

**Example:** If `traffic_congestion_index` shows `r = 0.68` with `actual_arrival_delay_min`, it's a strong candidate feature for Phase 7.

---

### Step 4.7: Target Variable Analysis
```r
class_pct <- df %>% count(delayed) %>%
  mutate(pct = round(n / sum(n) * 100, 1))

p19 <- ggplot(class_pct, aes(x = delayed, y = n, fill = delayed)) +
  geom_col(alpha = 0.8, width = 0.5) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), vjust = -0.5)
  # Displays both count and percentage on each bar
```

**Cross-tabulations exposing interactions:**
```r
# Does delayed rate differ significantly by weather? Season? Transport type?
print(table(df$delayed, df$weather_condition))
#         Clear  Cloudy  Fog  Rain  Snow  Storm
# 0 (No)   2100    1800  340   900   220     90
# 1 (Yes)   350     600  210   850   380    310
# Storm has ~78% delay rate vs Clear's ~14% — massive difference!
```

---

## Outputs (19+ files)
Key plots: `hour_weekday_heatmap.png`, `correlation_matrix.png`, `station_hour_heatmap.png`, `target_class_distribution.png`, all in `output/plots/`.

---

## 💡 Presentation Talking Points
> "The cross-tabulation between `delayed` and `weather_condition` revealed that Storm conditions have a 78% delay rate vs Clear's 14%. This 5:1 ratio validated our choice to create the `weather_severity_index` as a key interaction feature."

> "The correlation matrix immediately highlighted that `actual_departure_delay_min` and `actual_arrival_delay_min` are highly correlated — which is why we computed `delay_change` as the more informative derived feature."
