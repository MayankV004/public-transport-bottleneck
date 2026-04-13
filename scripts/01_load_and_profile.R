# =============================================================================
# PHASE 1: DATA LOADING & PROFILING
# Script: 01_load_and_profile.R
# Purpose: Load the public_transport_delays.csv dataset, inspect structure,
#          and generate a comprehensive data profiling report.
# =============================================================================

cat("=== PHASE 1: DATA LOADING & PROFILING ===\n\n")

# --- 1.0 Load required libraries ---------------------------------------------
library(readr)
library(dplyr)
library(tidyr)

# --- 1.1 Load the dataset ----------------------------------------------------
cat("Step 1.1: Loading dataset...\n")

df <- read_csv(
  "data/public_transport_delays.csv",
  col_types = cols(
    trip_id   = col_character(),
    route_id  = col_character(),
    date      = col_date(format = "%Y-%m-%d"),
    .default  = col_guess()
  ),
  show_col_types = FALSE
)

cat(sprintf("  Loaded %d rows x %d columns\n", nrow(df), ncol(df)))

# --- 1.2 Inspect structure ---------------------------------------------------
cat("\nStep 1.2: Dataset structure\n")
cat("--- str() ---\n")
str(df)

cat("\n--- glimpse() ---\n")
glimpse(df)

cat("\n--- summary() ---\n")
print(summary(df))

# --- 1.3 Data profiling report -----------------------------------------------
cat("\nStep 1.3: Generating profiling report...\n")

# Row and column count
profile_basic <- data.frame(
  Metric = c("Total Rows", "Total Columns"),
  Value  = c(nrow(df), ncol(df))
)

# Missing values per column
missing_counts <- colSums(is.na(df))
missing_pct    <- round(100 * missing_counts / nrow(df), 2)
missing_df <- data.frame(
  Column          = names(missing_counts),
  Missing_Count   = as.integer(missing_counts),
  Missing_Percent = missing_pct,
  stringsAsFactors = FALSE
)

cat("  Missing values per column:\n")
print(missing_df, row.names = FALSE)

# Unique values per column
unique_counts <- sapply(df, n_distinct)
unique_df <- data.frame(
  Column        = names(unique_counts),
  Unique_Values = as.integer(unique_counts),
  stringsAsFactors = FALSE
)

cat("\n  Unique values per column:\n")
print(unique_df, row.names = FALSE)

# Numeric column statistics
numeric_cols <- df %>% select(where(is.numeric))
if (ncol(numeric_cols) > 0) {
  numeric_stats <- numeric_cols %>%
    summarise(across(everything(), list(
      min    = ~min(., na.rm = TRUE),
      max    = ~max(., na.rm = TRUE),
      mean   = ~round(mean(., na.rm = TRUE), 2),
      median = ~round(median(., na.rm = TRUE), 2),
      sd     = ~round(sd(., na.rm = TRUE), 2)
    ))) %>%
    pivot_longer(everything(),
                 names_to  = c("Column", "Stat"),
                 names_sep = "_(?=[^_]+$)",
                 values_to = "Value") %>%
    pivot_wider(names_from = Stat, values_from = Value)
  
  cat("\n  Numeric column statistics:\n")
  print(as.data.frame(numeric_stats), row.names = FALSE)
}

# Categorical frequency tables
cat_cols <- c("transport_type", "weather_condition", "event_type", "season")
cat("\n  Categorical frequency tables:\n")
for (col in cat_cols) {
  if (col %in% names(df)) {
    cat(sprintf("\n  -- %s --\n", col))
    print(table(df[[col]]))
  }
}

# Date range
cat(sprintf("\n  Date range: %s to %s\n", min(df$date, na.rm = TRUE), max(df$date, na.rm = TRUE)))

# Class balance of target variable
cat("\n  Target variable (delayed) distribution:\n")
delay_table <- table(df$delayed)
delay_prop  <- round(prop.table(delay_table) * 100, 2)
print(delay_table)
cat("  Proportions: ")
print(delay_prop)

# --- 1.4 Initial observations ------------------------------------------------
cat("\nStep 1.4: Initial observations checks\n")

# Check for entirely-NA columns
all_na_cols <- names(df)[colSums(!is.na(df)) == 0]
if (length(all_na_cols) > 0) {
  cat(sprintf("  WARNING: Columns entirely NA: %s\n", paste(all_na_cols, collapse = ", ")))
} else {
  cat("  No columns are entirely NA.\n")
}

# Check origin == destination
circular <- sum(df$origin_station == df$destination_station, na.rm = TRUE)
cat(sprintf("  Circular trips (origin == destination): %d\n", circular))

# Check negative delay values (expected — means early)
neg_dep <- sum(df$actual_departure_delay_min < 0, na.rm = TRUE)
neg_arr <- sum(df$actual_arrival_delay_min < 0, na.rm = TRUE)
cat(sprintf("  Early departures (negative dep delay): %d\n", neg_dep))
cat(sprintf("  Early arrivals (negative arr delay): %d\n", neg_arr))

# Weekday encoding range
cat(sprintf("  Weekday range: %d to %d\n",
            min(df$weekday, na.rm = TRUE), max(df$weekday, na.rm = TRUE)))

# --- 1.5 Save profiling outputs ----------------------------------------------
cat("\nStep 1.5: Saving profiling outputs...\n")

# Combine profiling info into a single data frame
profile_report <- bind_rows(
  missing_df %>% mutate(Section = "Missing Values"),
  unique_df  %>% mutate(Section = "Unique Values")
)

write_csv(profile_report, "output/reports/data_profile.csv")
cat("  Saved: output/reports/data_profile.csv\n")

# Save the raw loaded data as an RDS for downstream scripts
saveRDS(df, "output/cleaned/raw_loaded.rds")
cat("  Saved: output/cleaned/raw_loaded.rds\n")

cat("\n=== PHASE 1 COMPLETE ===\n")
