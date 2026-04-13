# =============================================================================
# PHASE 5: DIMENSIONALITY REDUCTION (PCA)
# Script: 05_pca.R
# Purpose: Perform PCA on numeric features, determine optimal components,
#          generate scree/biplot visualizations, and save results.
# =============================================================================

cat("=== PHASE 5: DIMENSIONALITY REDUCTION (PCA) ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(FactoMineR)
library(factoextra)
library(ggplot2)

# --- Load transformed data ----------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
cat(sprintf("Loaded: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 5.1: Select Features for PCA
# =============================================================================
cat("Step 5.1: Selecting numeric features for PCA...\n")

pca_features <- c(
  # Delay metrics
  "actual_departure_delay_min", "actual_arrival_delay_min",
  "total_delay", "delay_change",
  # Weather metrics
  "temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
  # External factors
  "event_attendance_est", "traffic_congestion_index",
  # Derived features
  "weather_severity_index", "event_impact_score", "trip_duration_scheduled",
  # Pre-computed flags (numeric)
  "weekday"
)

# Filter to only columns that exist
pca_features <- pca_features[pca_features %in% names(df)]
cat(sprintf("  Selected %d features: %s\n", length(pca_features),
            paste(pca_features, collapse = ", ")))

pca_data <- df[, pca_features]
pca_data <- pca_data %>% mutate(across(everything(), as.numeric))

# Remove rows with any NA in PCA features
complete_rows <- complete.cases(pca_data)
pca_data <- pca_data[complete_rows, ]
cat(sprintf("  Complete cases for PCA: %d rows\n", nrow(pca_data)))

# =============================================================================
# Step 5.2: Standardize Features (Z-score)
# =============================================================================
cat("\nStep 5.2: Standardizing features...\n")

# Remove zero-variance columns
variances <- sapply(pca_data, var, na.rm = TRUE)
zero_var <- names(variances[variances == 0])
if (length(zero_var) > 0) {
  cat(sprintf("  Removing zero-variance columns: %s\n", paste(zero_var, collapse = ", ")))
  pca_data <- pca_data[, !names(pca_data) %in% zero_var]
}

pca_scaled <- scale(pca_data)

# Verify scaling
cat(sprintf("  Column means ≈ 0: %s\n",
            all(abs(colMeans(pca_scaled)) < 1e-10)))
cat(sprintf("  Column SDs ≈ 1: %s\n",
            all(abs(apply(pca_scaled, 2, sd) - 1) < 1e-10)))

# =============================================================================
# Step 5.3: Perform PCA
# =============================================================================
cat("\nStep 5.3: Performing PCA...\n")

pca_model <- prcomp(pca_scaled, center = FALSE, scale. = FALSE)

# Eigenvalues and variance explained
eigenvalues <- pca_model$sdev^2
var_explained <- round(eigenvalues / sum(eigenvalues) * 100, 2)
cum_var <- round(cumsum(var_explained), 2)

variance_table <- data.frame(
  Component          = paste0("PC", 1:length(eigenvalues)),
  Eigenvalue         = round(eigenvalues, 4),
  Variance_Pct       = var_explained,
  Cumulative_Pct     = cum_var
)

cat("\n  Variance explained by each component:\n")
print(variance_table, row.names = FALSE)

write_csv(variance_table, "output/reports/pca_variance_explained.csv")
cat("  Saved: output/reports/pca_variance_explained.csv\n")

# =============================================================================
# Step 5.4: Determine Number of Components
# =============================================================================
cat("\nStep 5.4: Determining optimal number of components...\n")

# Kaiser criterion: eigenvalue > 1
kaiser_k <- sum(eigenvalues > 1)
cat(sprintf("  Kaiser criterion (eigenvalue > 1): %d components\n", kaiser_k))

# Cumulative variance >= 85%
cum85_k <- which(cum_var >= 85)[1]
cat(sprintf("  Cumulative variance >= 85%%: %d components (%.1f%%)\n",
            cum85_k, cum_var[cum85_k]))

# Cumulative variance >= 90%
cum90_k <- which(cum_var >= 90)[1]
cat(sprintf("  Cumulative variance >= 90%%: %d components (%.1f%%)\n",
            cum90_k, cum_var[cum90_k]))

# Use the Kaiser criterion as default, but report all three
chosen_k <- kaiser_k
cat(sprintf("\n  CHOSEN: Retaining %d principal components (%.1f%% variance)\n",
            chosen_k, cum_var[chosen_k]))

# =============================================================================
# Step 5.5: Visualize PCA
# =============================================================================
cat("\nStep 5.5: Creating PCA visualizations...\n")

# Scree plot
p_scree <- fviz_eig(pca_model, addlabels = TRUE, ylim = c(0, 40),
                     barfill = "steelblue", barcolor = "steelblue") +
  labs(title = "PCA Scree Plot — Variance Explained per Component") +
  theme_minimal(base_size = 13)
ggsave("output/plots/pca_scree_plot.png", p_scree, width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/pca_scree_plot.png\n")

# Biplot (PC1 vs PC2)
p_biplot <- fviz_pca_biplot(pca_model,
                             repel = TRUE,
                             col.var = "red",
                             col.ind = "steelblue",
                             alpha.ind = 0.3,
                             label = "var") +
  labs(title = "PCA Biplot — PC1 vs PC2") +
  theme_minimal(base_size = 12)
ggsave("output/plots/pca_biplot.png", p_biplot, width = 12, height = 9, dpi = 300)
cat("  Saved: output/plots/pca_biplot.png\n")

# Variable contribution plot
p_var <- fviz_pca_var(pca_model,
                       col.var = "contrib",
                       gradient.cols = c("blue", "orange", "red"),
                       repel = TRUE) +
  labs(title = "PCA — Variable Contributions") +
  theme_minimal(base_size = 12)
ggsave("output/plots/pca_variable_contributions.png", p_var, width = 10, height = 8, dpi = 300)
cat("  Saved: output/plots/pca_variable_contributions.png\n")

# Loadings table
loadings_matrix <- pca_model$rotation[, 1:min(chosen_k, ncol(pca_model$rotation))]
loadings_df <- as.data.frame(round(loadings_matrix, 4))
loadings_df$Variable <- rownames(loadings_df)
loadings_df <- loadings_df[, c("Variable", setdiff(names(loadings_df), "Variable"))]
write_csv(loadings_df, "output/reports/pca_loadings.csv")
cat("  Saved: output/reports/pca_loadings.csv\n")

cat("\n  PCA Loadings (top components):\n")
print(loadings_df, row.names = FALSE)

# =============================================================================
# Step 5.6: Save PCA Outputs
# =============================================================================
cat("\nStep 5.6: Saving PCA outputs...\n")

# PCA-transformed data (selected components + identifiers)
pca_scores <- as.data.frame(pca_model$x[, 1:chosen_k])
pca_scores$trip_id <- df$trip_id[complete_rows]
pca_scores$delayed <- df$delayed[complete_rows]

# Move identifiers to the front
pca_scores <- pca_scores[, c("trip_id", "delayed",
                               paste0("PC", 1:chosen_k))]

write_csv(pca_scores, "output/cleaned/pca_transformed_data.csv")
cat("  Saved: output/cleaned/pca_transformed_data.csv\n")

# Save the PCA model
saveRDS(pca_model, "output/models/pca_model.rds")
cat("  Saved: output/models/pca_model.rds\n")

# Save number of chosen components for downstream scripts
saveRDS(list(k = chosen_k, complete_rows = complete_rows),
        "output/models/pca_config.rds")

cat(sprintf("\n  PCA Summary: Reduced %d features → %d components, retaining %.1f%% variance.\n",
            ncol(pca_scaled), chosen_k, cum_var[chosen_k]))

cat("\n=== PHASE 5 COMPLETE ===\n")
