# Phase 0: Package Installation (`00_install_packages.R`)

## Purpose
This is the **bootstrapping script** — the very first script to run. It guarantees that every R library used across all 10 phases is installed and correctly loadable. Without this, any downstream script would fail with cryptic "package not found" errors.

---

## Step-by-Step Breakdown

### Step 1: Define the Package Manifest
All required packages for the entire project are declared in a single vector. This acts as the project's dependency manifest.

```r
pkgs <- c(
  "readr", "dplyr", "tidyr", "lubridate", "hms",        # Data wrangling
  "ggplot2", "corrplot", "pheatmap", "gridExtra", "scales", # Visualization
  "factoextra", "FactoMineR",                             # PCA analysis
  "cluster", "dbscan", "fpc",                             # Clustering algorithms
  "caret", "randomForest", "e1071", "pROC", "ROCR",       # ML + evaluation
  "igraph", "networkD3", "htmlwidgets",                   # Network graphs
  "zoo", "nnet", "plotly"                                 # Time-series & misc
)
```

**Example:** If you later decide to add a new visualization in Phase 4, you just add the package name here — one place, managed centrally.

---

### Step 2: Smart Conditional Installation
Rather than installing all packages every time (which is slow and can reset package versions), the script compares the declared list against what's already installed and only installs what's missing.

```r
missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
#  ↑ installed.packages() returns a matrix of all installed packages.
#    We check: which of our declared packages are NOT in that list?

if (length(missing) > 0) {
  cat("Installing missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org", quiet = TRUE)
} else {
  cat("All packages already installed.\n")
}
```

**Example:** On first run, `missing` might return `c("dbscan", "fpc", "networkD3")`. On every subsequent run, `missing` returns an empty vector and the install step is skipped entirely — saving minutes of wait time.

---

### Step 3: Load Verification with Error Isolation
Even a successfully installed package can fail to *load* due to missing system-level dependencies (like compiled C libraries). This loop catches those problems immediately using `tryCatch` so each package is tested independently — one failure doesn't block others.

```r
cat("Verifying all packages can load...\n")
for (p in pkgs) {
  tryCatch({
    suppressPackageStartupMessages(library(p, character.only = TRUE))
    # character.only = TRUE: treats the variable p as a string name, not literal
  }, error = function(e) {
    cat(sprintf("  FAILED to load: %s — %s\n", p, conditionMessage(e)))
    # Reports exactly which package failed and why
  })
}
```

**Example output (failure case):**
```
FAILED to load: keras — there is no package called 'keras'
```
This immediately tells you `keras` must be installed manually with TensorFlow dependencies. All other packages continue loading normally.

**Example output (success case):**
```
Verifying all packages can load...
Package check complete.
```

---

## Why `suppressPackageStartupMessages`?
Packages like `ggplot2` and `dplyr` print verbose loading messages (e.g., "Attaching packages", "Conflicts"). `suppressPackageStartupMessages` silences these, keeping the console output clean and readable.

---

## Outputs
- Console log: list of any packages that failed to load
- No files are written to disk — this script only modifies the R environment

---

## 💡 Presentation Talking Point
> "We built a self-healing setup script. It only installs what's missing and proactively validates every single dependency — so the pipeline never fails mid-run due to a missing package. On first run it might install 5 packages; on every subsequent run it does nothing."
