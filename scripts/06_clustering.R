# =============================================================================
# PHASE 6: CLUSTERING (BOTTLENECK IDENTIFICATION)
# Script: 06_clustering.R
# Purpose: Determine optimal K, run K-Means and DBSCAN, profile clusters,
#          and identify delay-prone bottleneck groups.
# =============================================================================

cat("=== PHASE 6: CLUSTERING (BOTTLENECK IDENTIFICATION) ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(ggplot2)
library(cluster)
library(dbscan)
library(fpc)
library(factoextra)
library(gridExtra)

# --- Load PCA-transformed data ------------------------------------------------
pca_scores <- read_csv("output/cleaned/pca_transformed_data.csv", show_col_types = FALSE)
pca_config <- readRDS("output/models/pca_config.rds")
df <- readRDS("output/cleaned/transport_delays_transformed.rds")

chosen_k <- pca_config$k
complete_rows <- pca_config$complete_rows

cat(sprintf("Loaded PCA data: %d rows x %d PCs\n", nrow(pca_scores), chosen_k))

# Extract just the PC columns for clustering
pc_cols <- paste0("PC", 1:chosen_k)
cluster_data <- as.matrix(pca_scores[, pc_cols])

# =============================================================================
# Step 6.1: Prepare Clustering Input
# =============================================================================
cat("\nStep 6.1: Preparing clustering input...\n")

# Remove any remaining NAs
complete_mask <- complete.cases(cluster_data)
cluster_data <- cluster_data[complete_mask, ]
pca_scores_clean <- pca_scores[complete_mask, ]
cat(sprintf("  Clean clustering data: %d observations x %d PCs\n",
            nrow(cluster_data), ncol(cluster_data)))

# =============================================================================
# Step 6.2: Determine Optimal K (Elbow + Silhouette)
# =============================================================================
cat("\nStep 6.2: Determining optimal K...\n")

max_k <- 10
wss_values <- numeric(max_k)
sil_values <- numeric(max_k)

# Compute WSS and Silhouette for K = 2 to max_k
for (k in 2:max_k) {
  set.seed(42)
  km <- kmeans(cluster_data, centers = k, nstart = 25, iter.max = 100)
  wss_values[k] <- km$tot.withinss
  sil <- silhouette(km$cluster, dist(cluster_data))
  sil_values[k] <- mean(sil[, 3])
}
wss_values[1] <- sum(scale(cluster_data, scale = FALSE)^2)  # total SS for K=1

# Elbow plot
elbow_df <- data.frame(K = 1:max_k, WSS = wss_values)
p_elbow <- ggplot(elbow_df, aes(x = K, y = WSS)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "Elbow Method — Optimal K for K-Means",
       x = "Number of Clusters (K)", y = "Total Within-Cluster Sum of Squares") +
  scale_x_continuous(breaks = 1:max_k) +
  theme_minimal(base_size = 13)
ggsave("output/plots/kmeans_elbow_plot.png", p_elbow, width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/kmeans_elbow_plot.png\n")

# Silhouette plot
sil_df <- data.frame(K = 2:max_k, Silhouette = sil_values[2:max_k])
best_k_sil <- sil_df$K[which.max(sil_df$Silhouette)]

p_sil <- ggplot(sil_df, aes(x = K, y = Silhouette)) +
  geom_line(color = "coral", linewidth = 1.2) +
  geom_point(size = 3, color = "coral") +
  geom_vline(xintercept = best_k_sil, linetype = "dashed", color = "red") +
  labs(title = "Silhouette Analysis — Optimal K",
       subtitle = sprintf("Best K = %d (Avg Silhouette = %.3f)", best_k_sil,
                           max(sil_df$Silhouette)),
       x = "Number of Clusters (K)", y = "Average Silhouette Width") +
  scale_x_continuous(breaks = 2:max_k) +
  theme_minimal(base_size = 13)
ggsave("output/plots/kmeans_silhouette_analysis.png", p_sil, width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/kmeans_silhouette_analysis.png\n")

# Choose K (use silhouette-optimal, bounded between 2 and 5)
optimal_k <- max(2, min(best_k_sil, 5))
cat(sprintf("\n  Optimal K chosen: %d (Silhouette-based: %d)\n", optimal_k, best_k_sil))

# =============================================================================
# Step 6.3: Run K-Means Clustering
# =============================================================================
cat("\nStep 6.3: Running K-Means clustering with K=%d...\n", optimal_k)

set.seed(42)
km_model <- kmeans(cluster_data, centers = optimal_k, nstart = 25, iter.max = 100)

pca_scores_clean$cluster <- as.factor(km_model$cluster)
cat(sprintf("  Cluster sizes: %s\n",
            paste(table(km_model$cluster), collapse = ", ")))

# =============================================================================
# Step 6.4: Cluster Profiling
# =============================================================================
cat("\nStep 6.4: Profiling clusters...\n")

# Merge cluster labels back to the full (filtered) data
df_clustered <- df[complete_rows, ][complete_mask, ]
df_clustered$cluster <- km_model$cluster

# Compute profile metrics
cluster_profiles <- df_clustered %>%
  group_by(cluster) %>%
  summarise(
    n_observations       = n(),
    avg_arrival_delay    = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    avg_departure_delay  = round(mean(actual_departure_delay_min, na.rm = TRUE), 2),
    delay_rate           = round(mean(as.numeric(as.character(delayed)), na.rm = TRUE) * 100, 2),
    avg_congestion       = round(mean(traffic_congestion_index, na.rm = TRUE), 2),
    avg_temperature      = round(mean(temperature_C, na.rm = TRUE), 2),
    avg_precipitation    = round(mean(precipitation_mm, na.rm = TRUE), 2),
    dominant_transport   = names(which.max(table(transport_type))),
    dominant_weather     = names(which.max(table(weather_condition))),
    .groups = "drop"
  )

# Assign semantic labels based on delay_rate and avg_arrival_delay
cluster_profiles <- cluster_profiles %>%
  arrange(avg_arrival_delay) %>%
  mutate(label = case_when(
    row_number() == 1 ~ "On-Time / Stable",
    row_number() == n() ~ "Bottleneck-Prone / High-Delay",
    TRUE ~ "Moderate / Delay-Sensitive"
  ))

cat("\n  Cluster Profiles:\n")
print(as.data.frame(cluster_profiles), row.names = FALSE)

write_csv(cluster_profiles, "output/reports/cluster_profiles.csv")
cat("  Saved: output/reports/cluster_profiles.csv\n")

# =============================================================================
# Step 6.5: Visualize Clusters
# =============================================================================
cat("\nStep 6.5: Visualizing clusters...\n")

# PC1 vs PC2 coloured by cluster
p_cluster <- ggplot(pca_scores_clean, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  stat_ellipse(level = 0.95, linewidth = 1) +
  labs(title = sprintf("K-Means Clustering (K=%d) — PC1 vs PC2", optimal_k),
       x = "Principal Component 1", y = "Principal Component 2",
       color = "Cluster") +
  theme_minimal(base_size = 13) +
  scale_color_brewer(palette = "Set1")
ggsave("output/plots/cluster_pc1_pc2.png", p_cluster, width = 10, height = 8, dpi = 300)
cat("  Saved: output/plots/cluster_pc1_pc2.png\n")

# Cluster size bar chart
p_size <- ggplot(cluster_profiles, aes(x = factor(cluster), y = n_observations,
                                        fill = label)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = n_observations), vjust = -0.5, size = 4) +
  labs(title = "Cluster Sizes", x = "Cluster", y = "Number of Trips", fill = "Label") +
  theme_minimal(base_size = 13) +
  scale_fill_brewer(palette = "Set2")
ggsave("output/plots/cluster_sizes.png", p_size, width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/cluster_sizes.png\n")

# Cluster profile grouped bar chart
profile_long <- cluster_profiles %>%
  select(cluster, label, avg_arrival_delay, avg_departure_delay,
         avg_congestion, delay_rate) %>%
  tidyr::pivot_longer(cols = c(avg_arrival_delay, avg_departure_delay,
                                avg_congestion, delay_rate),
                       names_to = "Metric", values_to = "Value")

p_profile <- ggplot(profile_long, aes(x = Metric, y = Value, fill = factor(cluster))) +
  geom_col(position = "dodge", alpha = 0.8) +
  labs(title = "Cluster Profile Comparison",
       x = "Metric", y = "Value", fill = "Cluster") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")
ggsave("output/plots/cluster_profile_comparison.png", p_profile,
       width = 12, height = 7, dpi = 300)
cat("  Saved: output/plots/cluster_profile_comparison.png\n")

# =============================================================================
# Step 6.6: DBSCAN (Alternative Clustering)
# =============================================================================
cat("\nStep 6.6: Running DBSCAN...\n")

# k-distance plot to determine eps
png("output/plots/dbscan_knn_distance.png", width = 1000, height = 600, res = 150)
kNNdistplot(cluster_data, k = 5)
abline(h = 2, col = "red", lty = 2)
title("k-NN Distance Plot for DBSCAN eps Estimation")
dev.off()
cat("  Saved: output/plots/dbscan_knn_distance.png\n")

# Run DBSCAN with estimated eps
# Use median of 5-NN distances as a heuristic for eps
knn_dists <- kNNdist(cluster_data, k = 5)
eps_estimate <- median(sort(knn_dists)[round(0.9 * length(knn_dists))])
eps_estimate <- max(eps_estimate, 1.5)  # ensure reasonable lower bound

db_model <- dbscan::dbscan(cluster_data, eps = eps_estimate, minPts = 5)

cat(sprintf("  DBSCAN eps=%.2f, minPts=5\n", eps_estimate))
cat(sprintf("  DBSCAN clusters: %d (noise points = cluster 0: %d)\n",
            max(db_model$cluster), sum(db_model$cluster == 0)))
cat(sprintf("  DBSCAN cluster sizes: %s\n",
            paste(table(db_model$cluster), collapse = ", ")))

# DBSCAN scatter plot
pca_scores_clean$dbscan_cluster <- as.factor(db_model$cluster)

p_dbscan <- ggplot(pca_scores_clean, aes(x = PC1, y = PC2, color = dbscan_cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  labs(title = sprintf("DBSCAN Clustering (eps=%.2f) — PC1 vs PC2", eps_estimate),
       x = "Principal Component 1", y = "Principal Component 2",
       color = "Cluster") +
  theme_minimal(base_size = 13) +
  scale_color_brewer(palette = "Set1")
ggsave("output/plots/dbscan_cluster_plot.png", p_dbscan, width = 10, height = 8, dpi = 300)
cat("  Saved: output/plots/dbscan_cluster_plot.png\n")

# =============================================================================
# Step 6.7: Save Clustering Outputs
# =============================================================================
cat("\nStep 6.7: Saving clustering outputs...\n")

# Save clustered data
clustered_output <- pca_scores_clean
write_csv(clustered_output, "output/cleaned/clustered_data.csv")
cat("  Saved: output/cleaned/clustered_data.csv\n")

# Save models
saveRDS(km_model, "output/models/kmeans_model.rds")
saveRDS(db_model, "output/models/dbscan_model.rds")
cat("  Saved: output/models/kmeans_model.rds\n")
cat("  Saved: output/models/dbscan_model.rds\n")

# Create bottleneck flag for classification (Phase 7)
bottleneck_cluster <- cluster_profiles$cluster[cluster_profiles$label == "Bottleneck-Prone / High-Delay"]
df_clustered$bottleneck <- ifelse(df_clustered$cluster %in% bottleneck_cluster, 1, 0)
saveRDS(df_clustered, "output/cleaned/df_with_clusters.rds")
cat("  Saved: output/cleaned/df_with_clusters.rds\n")

cat(sprintf("\n  Clustering Summary: K-Means K=%d, DBSCAN found %d clusters.\n",
            optimal_k, max(db_model$cluster)))

cat("\n=== PHASE 6 COMPLETE ===\n")
