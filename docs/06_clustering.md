# Phase 6: Clustering — Bottleneck Identification (`06_clustering.R`)

## Purpose
Groups trips into behaviorally distinct segments using **K-Means** and **DBSCAN** clustering, operating on the PCA-compressed feature space from Phase 5. The primary deliverable is the **"Bottleneck-Prone"** cluster label — a data-driven identification of systemically delayed trips.

---

## Why Two Algorithms?

| | K-Means | DBSCAN |
|---|---|---|
| **Cluster Shape** | Spherical (circular) | Arbitrary shape |
| **Noise Handling** | Assigns every point to a cluster | Labels outliers as noise (cluster 0) |
| **Advantage** | Fast, interpretable, fixed K | No K needed, finds genuine density pockets |
| **Limitation** | Sensitive to initialisation | Sensitive to `eps` parameter |

Using both validates findings: agreement between algorithms increases confidence that the identified bottleneck cluster is real and not an artifact of one algorithm's assumptions.

---

## Step-by-Step Breakdown

### Step 6.1: Prepare Clustering Input
The PCA scores from Phase 5 (one row per trip, PC columns as features) are loaded and cleaned.

```r
pca_scores  <- read_csv("output/cleaned/pca_transformed_data.csv")
pca_config  <- readRDS("output/models/pca_config.rds")  # Contains chosen_k
df          <- readRDS("output/cleaned/transport_delays_transformed.rds")

chosen_k    <- pca_config$k        # Number of PCs to use (e.g., 7)
pc_cols     <- paste0("PC", 1:chosen_k)  # ["PC1", "PC2", ..., "PC7"]
cluster_data <- as.matrix(pca_scores[, pc_cols])
```

**Example dimensions:** `cluster_data` has shape `9,975 rows × 7 columns` — one row per trip, 7 PC axes.

---

### Step 6.2: Determine Optimal K — Elbow + Silhouette
Both metrics are computed for K = 2 to 10, giving us two independent guides to the best K.

```r
for (k in 2:max_k) {
  set.seed(42)  # Reproducibility
  km <- kmeans(cluster_data, centers = k, nstart = 25, iter.max = 100)

  wss_values[k] <- km$tot.withinss     # Total Within-Cluster Sum of Squares
  # Lower WSS = tighter clusters = better
  # But WSS always decreases as K increases, so we look for the "elbow"

  sil          <- silhouette(km$cluster, dist(cluster_data))
  sil_values[k] <- mean(sil[, 3])     # Average silhouette width
  # Range: -1 to 1. Higher = better separation between clusters
  # We choose K where silhouette is maximised
}
```

**Understanding WSS (Elbow Method):**
```
K=2: WSS = 8500   (big drop from K=1)
K=3: WSS = 6200   (medium drop)
K=4: WSS = 5100   (smaller drop)
K=5: WSS = 4800   (very small drop — "elbow" is here)
K=6: WSS = 4600   (diminishing returns)
→ Elbow at K=5 suggests 5 is the natural number of clusters
```

**Understanding Silhouette:**
```
K=2: Avg Silhouette = 0.31
K=3: Avg Silhouette = 0.38
K=4: Avg Silhouette = 0.41 ← MAXIMUM
K=5: Avg Silhouette = 0.36
→ Best K by silhouette = 4
Final chosen K = max(2, min(4, 5)) = 4  (bounded between 2 and 5)
```

---

### Step 6.3: K-Means Clustering
```r
set.seed(42)
km_model <- kmeans(cluster_data,
                   centers = optimal_k,   # e.g., 4
                   nstart  = 25,          # 25 random starts → picks best result
                   iter.max = 100)        # Max 100 iterations per start

pca_scores_clean$cluster <- as.factor(km_model$cluster)
cat(sprintf("Cluster sizes: %s\n", paste(table(km_model$cluster), collapse = ", ")))
```

**Example output:** `Cluster sizes: 3215, 2890, 2441, 1429`
This shows cluster 4 is the smallest — likely the bottleneck group (smallest because only a subset of trips are severely delayed).

---

### Step 6.4: Cluster Profiling & Semantic Labeling
The key step that transforms cluster numbers into actionable labels.

```r
cluster_profiles <- df_clustered %>%
  group_by(cluster) %>%
  summarise(
    n_observations      = n(),
    avg_arrival_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    avg_departure_delay = round(mean(actual_departure_delay_min, na.rm = TRUE), 2),
    delay_rate          = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    avg_congestion      = round(mean(traffic_congestion_index, na.rm = TRUE), 2),
    avg_temperature     = round(mean(temperature_C, na.rm = TRUE), 2),
    dominant_transport  = names(which.max(table(transport_type))),
    dominant_weather    = names(which.max(table(weather_condition))),
    .groups = "drop"
  )
```

**Example cluster profiles:**
```
Cluster  n_obs  avg_arr_delay  delay_rate  avg_congestion  dominant_weather  Label (assigned below)
      1   3215           3.1       18.5%            0.31           Clear      On-Time / Stable
      2   2890           7.4       44.2%            0.55           Cloudy      Moderate / Delay-Sensitive
      3   2441           9.8       58.7%            0.72           Rain        Moderate / Delay-Sensitive
      4   1429          19.3       83.1%            0.89           Storm       Bottleneck-Prone / High-Delay
```

```r
# Automatic semantic labeling: sorted by avg_arrival_delay
cluster_profiles <- cluster_profiles %>%
  arrange(avg_arrival_delay) %>%
  mutate(label = case_when(
    row_number() == 1    ~ "On-Time / Stable",             # LOWEST delay
    row_number() == n()  ~ "Bottleneck-Prone / High-Delay", # HIGHEST delay
    TRUE                 ~ "Moderate / Delay-Sensitive"     # Everything in between
  ))
```

---

### Step 6.5: Cluster Visualization
```r
# PC1 vs PC2 scatter — clusters shown with 95% confidence ellipses
ggplot(pca_scores_clean, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  stat_ellipse(level = 0.95, linewidth = 1) +  # 95% confidence ellipse per cluster
  scale_color_brewer(palette = "Set1")

# What to look for:
# Tight ellipses = compact, well-separated clusters (ideal)
# Overlapping ellipses = ambiguous boundary between clusters (expected at moderate delay levels)
```

---

### Step 6.6: DBSCAN — Density-Based Alternative
DBSCAN finds clusters as dense regions of data points — it doesn't require specifying K upfront.

```r
# Automatically estimate eps (the neighbourhood radius)
knn_dists    <- kNNdist(cluster_data, k = 5)  # Distance to 5th nearest neighbour
eps_estimate <- median(sort(knn_dists)[round(0.9 * length(knn_dists))])
# Takes the 90th percentile of 5-NN distances as the natural neighbourhood size

db_model <- dbscan::dbscan(cluster_data, eps = eps_estimate, minPts = 5)
# minPts=5: a region must have at least 5 points to be considered a dense cluster
```

**Example DBSCAN output:**
```
DBSCAN eps=2.14, minPts=5
DBSCAN clusters found: 3 (noise points = cluster 0: 412)
Cluster sizes:  412 (noise), 4823, 3190, 1550
```

**Key insight:** The 412 noise points (cluster 0) are statistical outliers — trips so unusual they don't fit any dense pattern. These warrant manual investigation for data quality or genuine network anomalies.

---

### Step 6.7: Bottleneck Flag for Phase 7
```r
# Create a binary feature: is this trip in the bottleneck cluster?
bottleneck_cluster <- cluster_profiles$cluster[
  cluster_profiles$label == "Bottleneck-Prone / High-Delay"
]
df_clustered$bottleneck <- ifelse(df_clustered$cluster %in% bottleneck_cluster, 1, 0)

# Example counts:
# bottleneck=0: 8,546 trips (normal)
# bottleneck=1: 1,429 trips (bottleneck-prone, 14.3% of dataset)
```

---

## Outputs
| File | Description |
|---|---|
| `output/reports/cluster_profiles.csv` | Per-cluster stats with semantic labels |
| `output/cleaned/clustered_data.csv` | Trip data with K-Means + DBSCAN labels |
| `output/cleaned/df_with_clusters.rds` | Full dataset with `bottleneck` flag for Phase 7 |
| `output/plots/cluster_pc1_pc2.png` | Scatter with 95% ellipses |
| `output/plots/kmeans_elbow_plot.png` | WSS elbow curve |
| `output/plots/kmeans_silhouette_analysis.png` | Silhouette score by K |
| `output/plots/dbscan_cluster_plot.png` | DBSCAN results scatter |

---

## 💡 Presentation Talking Points
> "Cluster 4 contains only 14.3% of all trips but accounts for 83% delay rate — this is our 'Bottleneck-Prone' group. Without clustering, this group would be lost in the average. With it, we can now ask: what do these 1,429 trips have in common? The answer guides operational decisions."

> "DBSCAN's 412 noise-labelled trips are equally interesting — they're statistical anomalies that don't belong to any dense pattern. These could be major incidents, sensor errors, or genuinely rare conditions worth investigating separately."
