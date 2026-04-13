# =============================================================================
# MAIN RUNNER SCRIPT
# run_all.R
# Purpose: Execute all project phases in sequence.
#          Run this from the project root directory.
#
# Usage:
#   setwd("d:/Sem 6/Data Warehousing Data Mining/Project")
#   source("scripts/run_all.R")
#
# Or run individual phases:
#   source("scripts/01_load_and_profile.R")
#   source("scripts/02_data_cleaning.R")
#   etc.
# =============================================================================

cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║  PUBLIC TRANSPORT DELAY ANALYSIS — FULL PIPELINE        ║\n")
cat("║  Data Warehousing & Data Mining Project                  ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n\n")

start_time <- Sys.time()

# --- Helper: Install packages if missing --------------------------------------
required_packages <- c(
  "readr", "dplyr", "tidyr", "lubridate", "hms", "zoo",
  "ggplot2", "corrplot", "pheatmap", "gridExtra", "scales",
  "factoextra", "FactoMineR",
  "cluster", "dbscan", "fpc",
  "caret", "randomForest", "e1071", "pROC", "ROCR",
  "igraph", "nnet"
)

cat("Checking and installing required packages...\n")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}
cat("All required packages available.\n\n")

# Optional packages (won't halt execution if missing)
optional_packages <- c("keras", "tensorflow", "networkD3", "htmlwidgets", "plotly")
for (pkg in optional_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Optional package '%s' not installed. Related features may use fallback.\n", pkg))
  }
}

# --- Ensure working directory is the project root ----------------------------
# Uncomment and adjust if needed:
# setwd("d:/Sem 6/Data Warehousing Data Mining/Project")

cat(sprintf("\nWorking directory: %s\n\n", getwd()))

# --- Create directories if they don't exist -----------------------------------
dirs <- c("data", "output/cleaned", "output/plots", "output/models", "output/reports", "scripts")
for (d in dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# =============================================================================
# EXECUTE PHASES
# =============================================================================

run_phase <- function(script_name, phase_name) {
  cat(sprintf("\n%s\n", paste(rep("=", 60), collapse = "")))
  cat(sprintf(">>> Starting %s...\n", phase_name))
  cat(sprintf("%s\n\n", paste(rep("=", 60), collapse = "")))
  
  phase_start <- Sys.time()
  
  tryCatch({
    source(file.path("scripts", script_name), local = FALSE)
    elapsed <- round(difftime(Sys.time(), phase_start, units = "secs"), 1)
    cat(sprintf("\n>>> %s completed in %.1f seconds.\n", phase_name, elapsed))
  }, error = function(e) {
    cat(sprintf("\n>>> ERROR in %s: %s\n", phase_name, conditionMessage(e)))
    cat(">>> Continuing with next phase...\n")
  })
}

# Phase 1: Data Loading & Profiling
run_phase("01_load_and_profile.R", "Phase 1: Data Loading & Profiling")

# Phase 2: Data Cleaning
run_phase("02_data_cleaning.R", "Phase 2: Data Cleaning")

# Phase 3: Data Transformation & Feature Engineering
run_phase("03_transformation.R", "Phase 3: Data Transformation")

# Phase 4: Exploratory Data Analysis
run_phase("04_eda.R", "Phase 4: EDA")

# Phase 5: PCA (Dimensionality Reduction)
run_phase("05_pca.R", "Phase 5: PCA")

# Phase 6: Clustering
run_phase("06_clustering.R", "Phase 6: Clustering")

# Phase 7: ML Classification
run_phase("07_classification.R", "Phase 7: Classification")

# Phase 8: LSTM Forecasting
run_phase("08_lstm.R", "Phase 8: LSTM")

# Phase 9: Network Analysis
run_phase("09_propagation.R", "Phase 9: Network Analysis")

# Phase 10: Reporting & Power BI Exports
run_phase("10_reporting.R", "Phase 10: Reporting")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

total_time <- round(difftime(Sys.time(), start_time, units = "mins"), 1)

cat("\n\n╔══════════════════════════════════════════════════════════╗\n")
cat("║  PIPELINE EXECUTION COMPLETE                              ║\n")
cat(sprintf("║  Total time: %.1f minutes                                 ║\n", total_time))
cat("╚══════════════════════════════════════════════════════════╝\n\n")

# List all generated outputs
cat("Generated outputs:\n")
all_files <- list.files("output", recursive = TRUE)
cat(sprintf("  Total files: %d\n", length(all_files)))
cat(sprintf("  Cleaned data: %d files\n", length(list.files("output/cleaned"))))
cat(sprintf("  Plots: %d files\n", length(list.files("output/plots"))))
cat(sprintf("  Models: %d files\n", length(list.files("output/models"))))
cat(sprintf("  Reports: %d files\n", length(list.files("output/reports"))))

cat("\nDone! Review the output/ folder for all deliverables.\n")
