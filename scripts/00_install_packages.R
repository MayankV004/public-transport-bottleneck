pkgs <- c(
  "readr", "dplyr", "tidyr", "lubridate", "hms",
  "ggplot2", "corrplot", "pheatmap", "gridExtra", "scales",
  "factoextra", "FactoMineR", "cluster", "dbscan", "fpc",
  "caret", "randomForest", "e1071", "pROC", "ROCR",
  "igraph", "networkD3", "htmlwidgets", "zoo", "nnet", "plotly"
)

missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]

if (length(missing) > 0) {
  cat("Installing missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org", quiet = TRUE)
  cat("Done installing.\n")
} else {
  cat("All packages already installed.\n")
}

cat("Verifying all packages can load...\n")
for (p in pkgs) {
  tryCatch({
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }, error = function(e) {
    cat(sprintf("  FAILED to load: %s — %s\n", p, conditionMessage(e)))
  })
}
cat("Package check complete.\n")
