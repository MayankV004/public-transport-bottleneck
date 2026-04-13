# Master Runner (`run_all.R`)

## Purpose
Executes all 10 phases in sequence from a single entry point. Designed for **one-command reproducibility** — any collaborator can rebuild all outputs from scratch by running this single file. Also demonstrates professional pipeline design with dependency checks, structured logging, and graceful error recovery.

---

## Step-by-Step Breakdown

### Step 1: Project Banner
The script opens with a visual banner in the console, making it immediately clear in logs what ran and when.

```r
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║  PUBLIC TRANSPORT DELAY ANALYSIS — FULL PIPELINE        ║\n")
cat("║  Data Warehousing & Data Mining Project                  ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n\n")

start_time <- Sys.time()   # Record start time for total duration reporting
```

---

### Step 2: Smart Package Installation
Checks and installs only missing packages, split into *required* (must succeed) and *optional* (may fail without halting the pipeline).

```r
# Required: pipeline cannot run without these
required_packages <- c("readr", "dplyr", "tidyr", "lubridate", "hms", "zoo",
                        "ggplot2", "corrplot", "pheatmap", "gridExtra", "scales",
                        "factoextra", "FactoMineR", "cluster", "dbscan", "fpc",
                        "caret", "randomForest", "e1071", "pROC", "ROCR",
                        "igraph", "nnet")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}
cat("All required packages available.\n\n")

# Optional: scripts have fallback logic for these
optional_packages <- c("keras", "tensorflow", "networkD3", "htmlwidgets", "plotly")
for (pkg in optional_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Optional '%s' not installed. Related features will use fallback.\n", pkg))
  }
}
```

**Example console output on first run:**
```
Installing dbscan...
Installing fpc...
All required packages available.

  Optional 'keras' not installed. Related features will use fallback.
  Optional 'networkD3' not installed. Related features will use fallback.
```

---

### Step 3: Directory Structure Setup
Creates all required output directories if they don't already exist. This means cloning the project to a new machine and running immediately works without manual setup.

```r
dirs <- c("data", "output/cleaned", "output/plots", "output/models", "output/reports", "scripts")
for (d in dirs) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)   # recursive=TRUE creates nested dirs (like mkdir -p)
    cat(sprintf("  Created directory: %s\n", d))
  }
}
cat(sprintf("Working directory: %s\n\n", getwd()))
```

**Example output (first run in empty project):**
```
Created directory: output/cleaned
Created directory: output/plots
Created directory: output/models
Created directory: output/reports
Working directory: /home/user/public-transport-bottleneck
```

---

### Step 4: Fault-Tolerant Phase Runner
The `run_phase()` function is the core design pattern: each phase runs inside `tryCatch`, so one phase's failure cannot prevent subsequent phases from running.

```r
run_phase <- function(script_name, phase_name) {
  # Print a clear separator before each phase
  cat(sprintf("\n%s\n", paste(rep("=", 60), collapse = "")))
  cat(sprintf(">>> Starting %s...\n", phase_name))
  cat(sprintf("%s\n\n", paste(rep("=", 60), collapse = "")))

  phase_start <- Sys.time()

  tryCatch({
    source(file.path("scripts", script_name), local = FALSE)
    # local=FALSE: runs in the global environment, so df and models are accessible later

    elapsed <- round(difftime(Sys.time(), phase_start, units = "secs"), 1)
    cat(sprintf("\n>>> %s completed in %.1f seconds.\n", phase_name, elapsed))

  }, error = function(e) {
    cat(sprintf("\n>>> ERROR in %s: %s\n", phase_name, conditionMessage(e)))
    cat(">>> Continuing with next phase...\n")
    # The pipeline does NOT halt — subsequent phases still run
  })
}
```

**Example: Failure in Phase 8 (LSTM) due to Keras being unavailable:**
```
============================================================
>>> Starting Phase 8: LSTM...
============================================================
  NOTE: keras/tensorflow not functional. Using nnet fallback.
  [nnet model trains and evaluates successfully]
>>> Phase 8: LSTM completed in 18.3 seconds.
```

**Example: Failure in Phase 6 (Clustering) due to corrupted PCA output:**
```
>>> ERROR in Phase 6: Clustering: cannot open file 'output/cleaned/pca_transformed_data.csv'
>>> Continuing with next phase...
============================================================
>>> Starting Phase 7: Classification...   ← Pipeline continues!
```

---

### Step 5: Phase Execution Sequence
Each phase is called in dependency order — clean data needed before transformation, transformation needed before EDA, etc.

```r
run_phase("01_load_and_profile.R",  "Phase 1: Data Loading & Profiling")
run_phase("02_data_cleaning.R",     "Phase 2: Data Cleaning")
run_phase("03_transformation.R",    "Phase 3: Data Transformation")
run_phase("04_eda.R",               "Phase 4: EDA")
run_phase("05_pca.R",               "Phase 5: PCA")
run_phase("06_clustering.R",        "Phase 6: Clustering")
run_phase("07_classification.R",    "Phase 7: Classification")
run_phase("08_lstm.R",              "Phase 8: LSTM")
run_phase("09_propagation.R",       "Phase 9: Network Analysis")
run_phase("10_reporting.R",         "Phase 10: Reporting")
```

**Dependency chain:**
```
01 → 02 → 03 → (04, 05)
               05 → 06 → 07
               03 → 08
               06 → 09
               03 → 10
```

---

### Step 6: Final Summary Report
After all phases complete, a structured summary is printed including total elapsed time and output file counts.

```r
total_time <- round(difftime(Sys.time(), start_time, units = "mins"), 1)

cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║  PIPELINE EXECUTION COMPLETE                              ║\n")
cat(sprintf("║  Total time: %.1f minutes                                 ║\n", total_time))
cat("╚══════════════════════════════════════════════════════════╝\n\n")

cat(sprintf("Generated outputs:\n"))
cat(sprintf("  Total files: %d\n",   length(list.files("output", recursive = TRUE))))
cat(sprintf("  Cleaned data: %d\n",  length(list.files("output/cleaned"))))
cat(sprintf("  Plots: %d\n",         length(list.files("output/plots"))))
cat(sprintf("  Models: %d\n",        length(list.files("output/models"))))
cat(sprintf("  Reports: %d\n",       length(list.files("output/reports"))))
```

**Example final output:**
```
╔══════════════════════════════════════════════════════════╗
║  PIPELINE EXECUTION COMPLETE                              ║
║  Total time: 12.4 minutes                                 ║
╚══════════════════════════════════════════════════════════╝

Generated outputs:
  Total files: 47
  Cleaned data: 8
  Plots: 24
  Models: 9
  Reports: 6

Done! Review the output/ folder for all deliverables.
```

---

## How to Run

```r
# Option 1: Run from within R (set project root as working directory first)
setwd("/home/streamliner/public-transport-bottleneck")
source("scripts/run_all.R")

# Option 2: Run a single phase for debugging
source("scripts/03_transformation.R")

# Option 3: Run from terminal (Rscript)
# Rscript scripts/run_all.R
```

---

## 💡 Presentation Talking Points
> "The entire pipeline — from raw CSV to Power BI exports, trained neural networks, and network graphs — runs in a single command and completes in about 12 minutes. This is the standard we aimed for: push-button reproducibility."

> "The fault-tolerant runner was a deliberate design choice. In real projects, Phase 8 (deep learning) might fail on a machine without a GPU. Rather than stopping the entire pipeline, we log the error and continue — all other 9 phases still produce their outputs."
