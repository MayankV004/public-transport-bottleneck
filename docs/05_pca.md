# Phase 5: Dimensionality Reduction — PCA (`05_pca.R`)

## Purpose
Reduces 60+ engineered features into a compact set of **Principal Components** that capture most of the dataset's variance. This serves two purposes: (1) removes multicollinearity before clustering, and (2) produces a 2D/3D view of the data for visualization.

---

## Why PCA Before Clustering?

**The problem with raw features in K-Means:**
Many features are highly correlated. For example:
- `actual_arrival_delay_min` and `total_delay` are nearly identical
- `temperature_C` and `feels_like_impact` overlap significantly

When K-Means sees correlated features, they get "double-counted", distorting cluster shapes. PCA collapses correlated groups into single orthogonal axes (Principal Components), giving K-Means a cleaner input.

---

## Step-by-Step Breakdown

### Step 5.1: Select Features for PCA
Only numeric, *pre-trip-knowable* features are included. Target variables and encoded categoricals with low information are excluded.

```r
pca_features <- c(
  # Delay metrics (outcome-side features)
  "actual_departure_delay_min", "actual_arrival_delay_min", "total_delay", "delay_change",
  # Weather (pre-trip forecast)
  "temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
  # External context
  "event_attendance_est", "traffic_congestion_index",
  # Derived composite features (from Phase 3)
  "weather_severity_index", "event_impact_score", "trip_duration_scheduled",
  "weekday"
)

# Safety check: only keep features that actually exist in the dataset
pca_features <- pca_features[pca_features %in% names(df)]
cat(sprintf("Selected %d features for PCA\n", length(pca_features)))
```

**Example output:** `Selected 14 features for PCA`

---

### Step 5.2: Standardize Features
PCA is purely distance-based. A feature measured in thousands (e.g., `event_attendance_est = 50,000`) would completely dominate one measured in single digits (e.g., `weekday = 3`) without standardization.

```r
# Remove zero-variance columns first — they contribute no information
variances <- sapply(pca_data, var, na.rm = TRUE)
zero_var  <- names(variances[variances == 0])
if (length(zero_var) > 0) {
  cat(sprintf("Removing zero-variance columns: %s\n", paste(zero_var, collapse = ", ")))
  pca_data <- pca_data[, !names(pca_data) %in% zero_var]
}

pca_scaled <- scale(pca_data)  # Each column: subtract mean, divide by SD

# Verify scaling worked
cat(sprintf("Column means ≈ 0: %s\n", all(abs(colMeans(pca_scaled)) < 1e-10)))
cat(sprintf("Column SDs   ≈ 1: %s\n", all(abs(apply(pca_scaled, 2, sd) - 1) < 1e-10)))
```

**Example output:**
```
Column means ≈ 0: TRUE
Column SDs   ≈ 1: TRUE
```

**Scaling example:**
```
actual_arrival_delay_min before scaling: [-5, 0, 4, 12, 27]   range = 32
After Z-score scaling:                  [-1.12, -0.68, -0.32, 0.39, 1.73] range = 2.85
event_attendance_est before: [1000, 5000, 25000, 80000]  range = 79,000
After Z-score scaling:        [-0.9, -0.7, 0.1, 1.5]    range = 2.4
→ Now both features are on the same scale
```

---

### Step 5.3: Perform PCA
`prcomp` computes the principal components. The output `$rotation` matrix explains *what* each PC represents.

```r
pca_model <- prcomp(pca_scaled, center = FALSE, scale. = FALSE)
# center=FALSE and scale.=FALSE because we already scaled in Step 5.2

# Compute variance explained by each PC
eigenvalues  <- pca_model$sdev^2             # Variance = squared standard deviation
var_explained <- round(eigenvalues / sum(eigenvalues) * 100, 2)
cum_var       <- round(cumsum(var_explained), 2)
```

**Example variance table output:**
```
Component  Eigenvalue  Variance_Pct  Cumulative_Pct
      PC1       3.821         27.29           27.29
      PC2       2.143         15.31           42.60
      PC3       1.876         13.40           56.00
      PC4       1.204          8.60           64.60
      PC5       1.051          7.51           72.11
      PC6       0.892          6.37           78.48
      PC7       0.741          5.29           83.77
      PC8       0.650          4.64           88.41
```

---

### Step 5.4: Determining the Optimal Number of Components

Three standard criteria are evaluated:

```r
# 1. Kaiser Criterion: keep all PCs with eigenvalue > 1
#    (Eigenvalue > 1 means the PC explains more variance than a single raw variable)
kaiser_k <- sum(eigenvalues > 1)
cat(sprintf("Kaiser criterion: %d components\n", kaiser_k))

# 2. Minimum threshold: at least 85% cumulative variance
cum85_k <- which(cum_var >= 85)[1]
cat(sprintf("Cumulative variance >= 85%%: %d components (%.1f%%)\n", cum85_k, cum_var[cum85_k]))

# 3. Conservative threshold: at least 90% cumulative variance
cum90_k <- which(cum_var >= 90)[1]
cat(sprintf("Cumulative variance >= 90%%: %d components (%.1f%%)\n", cum90_k, cum_var[cum90_k]))
```

**Example output:**
```
Kaiser criterion: 7 components
Cumulative variance >= 85%: 8 components (88.41%)
Cumulative variance >= 90%: 9 components (92.15%)
CHOSEN: Retaining 7 principal components (83.77% variance)
```

**Decision logic:** Kaiser is chosen by default as the most conservative interpretable rule — we keep only PCs that "pay for themselves" by explaining more variance than a single original variable.

---

### Step 5.5: Visualizations

**Scree Plot:** Shows diminishing returns — how much each additional PC contributes. Look for the "elbow" where the curve flattens.
```r
p_scree <- fviz_eig(pca_model, addlabels = TRUE, ylim = c(0, 40))
# Labels show the % variance above each bar
```

**Biplot (PC1 vs PC2):** Shows both data points (blue dots) and feature arrows (red) in the same space.
```r
p_biplot <- fviz_pca_biplot(pca_model,
  repel = TRUE,         # Prevent label overlap
  col.var = "red",      # Feature arrows in red
  col.ind = "steelblue", alpha.ind = 0.3,  # Samples in semi-transparent blue
  label = "var")        # Label only the feature arrows
```

**Reading a biplot:**
- Features with arrows pointing in the same direction are positively correlated
- Features with arrows pointing opposite are negatively correlated
- Features pointing perpendicular have no linear correlation
- Data points in the direction of an arrow have high values on that feature

**Example insight:** If `precipitation_mm` and `delay_change` arrows point in the same direction, rain correlates with delay growing during trips.

---

### Step 5.6: Save PCA Scores for Clustering
```r
# Extract the chosen component scores for every trip
pca_scores <- as.data.frame(pca_model$x[, 1:chosen_k])
pca_scores$trip_id <- df$trip_id[complete_rows]
pca_scores$delayed <- df$delayed[complete_rows]

write_csv(pca_scores, "output/cleaned/pca_transformed_data.csv")
cat(sprintf("Reduced %d features → %d components, retaining %.1f%% variance.\n",
            ncol(pca_scaled), chosen_k, cum_var[chosen_k]))
```

**Example output:** `Reduced 14 features → 7 components, retaining 83.77% variance.`

---

## Outputs
| File | Description |
|---|---|
| `output/cleaned/pca_transformed_data.csv` | Trips expressed as PC scores |
| `output/models/pca_model.rds` | Fitted model (for applying to new data) |
| `output/reports/pca_variance_explained.csv` | Eigenvalue and % variance per PC |
| `output/reports/pca_loadings.csv` | Feature contribution to each PC |
| `output/plots/pca_scree_plot.png` | Scree plot |
| `output/plots/pca_biplot.png` | Biplot (PC1 vs PC2) |

---

## 💡 Presentation Talking Points
> "After PCA, 14 correlated features become 7 orthogonal (uncorrelated) components. This isn't just data compression — it's mathematical denoising. K-Means on the PC scores finds cleaner, more meaningful clusters than it would on raw features."

> "The biplot showed us that PC1 is essentially a 'delay severity' axis — all delay-related features (total_delay, actual_arrival_delay, severe_delay) load heavily on it. PC2 captures weather severity. This gives our components human-interpretable meaning."
