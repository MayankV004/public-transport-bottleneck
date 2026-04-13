# =============================================================================
# PHASE 8: DEEP LEARNING — LSTM DELAY FORECASTING
# Script: 08_lstm.R
# Purpose: Build an LSTM neural network to forecast hourly average delay
#          from a time-series of aggregated delay data.
# =============================================================================

cat("=== PHASE 8: DEEP LEARNING — LSTM DELAY FORECASTING ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

# Check if keras is available; if not, use a fallback approach
keras_available <- tryCatch({
  library(keras)
  # Test if tensorflow is actually functional
  tf <- tensorflow::tf
  tf$constant(1)
  TRUE
}, error = function(e) {
  cat("  NOTE: keras/tensorflow not functional. Using nnet fallback.\n")
  FALSE
})

# --- Load transformed data ----------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
cat(sprintf("Loaded: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 8.1: Create Time-Series Dataset
# =============================================================================
cat("Step 8.1: Creating time-series dataset...\n")

# Sort by date + time
df <- df %>% arrange(date, time)

# Aggregate to hourly average arrival delay
hourly_ts <- df %>%
  group_by(date, hour) %>%
  summarise(
    avg_delay     = mean(actual_arrival_delay_min, na.rm = TRUE),
    trip_count    = n(),
    avg_congestion = mean(traffic_congestion_index, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(date, hour)

cat(sprintf("  Hourly time series: %d data points\n", nrow(hourly_ts)))

# Fill missing hours with forward fill / interpolation
all_dates <- seq(min(hourly_ts$date), max(hourly_ts$date), by = "day")
all_hours <- expand.grid(date = all_dates, hour = 0:23) %>%
  arrange(date, hour)

hourly_ts <- all_hours %>%
  left_join(hourly_ts, by = c("date", "hour"))

# Forward-fill NAs
hourly_ts$avg_delay <- zoo::na.locf(hourly_ts$avg_delay, na.rm = FALSE)
hourly_ts$avg_delay[is.na(hourly_ts$avg_delay)] <- mean(hourly_ts$avg_delay, na.rm = TRUE)

write_csv(hourly_ts, "output/cleaned/hourly_delay_timeseries.csv")
cat("  Saved: output/cleaned/hourly_delay_timeseries.csv\n")

# =============================================================================
# Step 8.2: Normalize the Time Series
# =============================================================================
cat("\nStep 8.2: Normalizing time series (Min-Max [0,1])...\n")

delay_series <- hourly_ts$avg_delay
min_delay <- min(delay_series, na.rm = TRUE)
max_delay <- max(delay_series, na.rm = TRUE)
range_delay <- max_delay - min_delay

if (range_delay == 0) range_delay <- 1  # prevent division by zero

delay_scaled <- (delay_series - min_delay) / range_delay
cat(sprintf("  Min delay: %.2f, Max delay: %.2f\n", min_delay, max_delay))

# Save normalization params
norm_params <- list(min_delay = min_delay, max_delay = max_delay, range_delay = range_delay)
saveRDS(norm_params, "output/models/lstm_normalization_params.rds")

# =============================================================================
# Step 8.3: Create Sequences (Sliding Window)
# =============================================================================
cat("\nStep 8.3: Creating sliding window sequences...\n")

lookback <- 24  # Use last 24 hours to predict next hour

# If we don't have enough data points, reduce lookback
if (length(delay_scaled) < lookback + 10) {
  lookback <- max(4, floor(length(delay_scaled) / 3))
  cat(sprintf("  Adjusted lookback to %d (limited data points)\n", lookback))
}

n <- length(delay_scaled)
n_sequences <- n - lookback

if (n_sequences < 10) {
  cat("  WARNING: Very few sequences. Results may be unreliable.\n")
}

# Create X and y
X <- matrix(0, nrow = n_sequences, ncol = lookback)
y <- numeric(n_sequences)

for (i in 1:n_sequences) {
  X[i, ] <- delay_scaled[i:(i + lookback - 1)]
  y[i]   <- delay_scaled[i + lookback]
}

# Reshape for LSTM: (samples, timesteps, features)
X_lstm <- array(X, dim = c(nrow(X), lookback, 1))

cat(sprintf("  Sequences created: %d samples, lookback = %d\n", n_sequences, lookback))

# =============================================================================
# Step 8.4: Train-Test Split (Sequential)
# =============================================================================
cat("\nStep 8.4: Sequential train-test split (80/20)...\n")

split_idx <- floor(0.8 * n_sequences)
X_train <- X_lstm[1:split_idx, , , drop = FALSE]
y_train <- y[1:split_idx]
X_test  <- X_lstm[(split_idx + 1):n_sequences, , , drop = FALSE]
y_test  <- y[(split_idx + 1):n_sequences]

cat(sprintf("  Training: %d sequences, Test: %d sequences\n",
            length(y_train), length(y_test)))

# =============================================================================
# Steps 8.5–8.8: Build, Train, Evaluate LSTM
# =============================================================================

if (keras_available) {
  # ------- KERAS LSTM APPROACH -------
  cat("\nStep 8.5: Building LSTM model (Keras)...\n")
  
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(lookback, 1)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 25, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = "adam",
    loss      = "mse",
    metrics   = c("mae")
  )
  
  cat("  Model architecture:\n")
  summary(model)
  
  cat("\nStep 8.6: Training LSTM...\n")
  
  history <- model %>% fit(
    X_train, y_train,
    epochs          = 50,
    batch_size      = 32,
    validation_split = 0.2,
    callbacks       = list(
      callback_early_stopping(patience = 10, restore_best_weights = TRUE)
    ),
    verbose = 1
  )
  
  # Training loss curve
  png("output/plots/lstm_training_history.png", width = 1000, height = 600, res = 150)
  plot(history)
  dev.off()
  cat("  Saved: output/plots/lstm_training_history.png\n")
  
  # Predict
  cat("\nStep 8.7: Evaluating LSTM...\n")
  y_pred_scaled <- model %>% predict(X_test)
  y_pred_scaled <- as.vector(y_pred_scaled)
  
  # Save model
  tryCatch({
    model %>% save_model_hdf5("output/models/lstm_model.h5")
    cat("  Saved: output/models/lstm_model.h5\n")
  }, error = function(e) {
    cat("  Could not save .h5 model:", conditionMessage(e), "\n")
    saveRDS(model, "output/models/lstm_model.rds")
    cat("  Saved: output/models/lstm_model.rds (RDS fallback)\n")
  })
  
} else {
  # ------- FALLBACK: Simple neural-network style approach with nnet -------
  cat("\nStep 8.5 (Fallback): Building simple NN model...\n")
  
  if (!requireNamespace("nnet", quietly = TRUE)) {
    install.packages("nnet", repos = "https://cloud.r-project.org")
  }
  library(nnet)
  
  # Flatten X for nnet
  X_train_flat <- matrix(X_train, nrow = dim(X_train)[1], ncol = lookback)
  X_test_flat  <- matrix(X_test,  nrow = dim(X_test)[1],  ncol = lookback)
  
  train_nn <- as.data.frame(cbind(X_train_flat, y = y_train))
  
  set.seed(42)
  nn_model <- nnet(y ~ ., data = train_nn, size = 25, linout = TRUE,
                   maxit = 200, trace = FALSE)
  
  test_nn <- as.data.frame(X_test_flat)
  names(test_nn) <- names(train_nn)[1:lookback]
  y_pred_scaled <- predict(nn_model, test_nn)
  y_pred_scaled <- as.vector(y_pred_scaled)
  
  saveRDS(nn_model, "output/models/lstm_model_fallback.rds")
  cat("  Saved: output/models/lstm_model_fallback.rds\n")
}

# =============================================================================
# Step 8.7 (continued): Inverse-transform & Compute Metrics
# =============================================================================

# Inverse transform
y_actual <- y_test * range_delay + min_delay
y_pred   <- y_pred_scaled * range_delay + min_delay

# Metrics
mse_val  <- mean((y_actual - y_pred)^2)
rmse_val <- sqrt(mse_val)
mae_val  <- mean(abs(y_actual - y_pred))
mape_val <- mean(abs((y_actual - y_pred) / ifelse(y_actual == 0, 1, y_actual))) * 100

metrics_df <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "MAPE"),
  Value  = round(c(mse_val, rmse_val, mae_val, mape_val), 4)
)

cat("\n  LSTM Performance Metrics:\n")
print(metrics_df, row.names = FALSE)

write_csv(metrics_df, "output/reports/lstm_performance_metrics.csv")
cat("  Saved: output/reports/lstm_performance_metrics.csv\n")

# =============================================================================
# Step 8.8: Visualizations
# =============================================================================
cat("\nStep 8.8: Creating LSTM visualizations...\n")

result_df <- data.frame(
  Index    = 1:length(y_actual),
  Actual   = y_actual,
  Predicted = y_pred
)

# Actual vs Predicted time series
p_ts <- ggplot(result_df, aes(x = Index)) +
  geom_line(aes(y = Actual), color = "steelblue", linewidth = 0.8) +
  geom_line(aes(y = Predicted), color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(title = "LSTM Forecasting: Actual vs Predicted Hourly Delay",
       subtitle = sprintf("RMSE = %.2f min | MAE = %.2f min", rmse_val, mae_val),
       x = "Time Index (hours)", y = "Avg Arrival Delay (min)") +
  theme_minimal(base_size = 13) +
  annotate("text", x = Inf, y = Inf, label = "Blue = Actual, Red = Predicted",
           hjust = 1.1, vjust = 1.5, size = 3.5, color = "gray40")
ggsave("output/plots/lstm_actual_vs_predicted.png", p_ts, width = 12, height = 6, dpi = 300)
cat("  Saved: output/plots/lstm_actual_vs_predicted.png\n")

# Scatter: Actual vs Predicted
p_scatter <- ggplot(result_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1, linetype = "dashed") +
  labs(title = "LSTM: Actual vs Predicted Scatter",
       x = "Actual Delay (min)", y = "Predicted Delay (min)") +
  theme_minimal(base_size = 13)
ggsave("output/plots/lstm_scatter_actual_vs_predicted.png", p_scatter,
       width = 8, height = 7, dpi = 300)
cat("  Saved: output/plots/lstm_scatter_actual_vs_predicted.png\n")

# Residual distribution
result_df$Residual <- result_df$Actual - result_df$Predicted
p_resid <- ggplot(result_df, aes(x = Residual)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(xintercept = 0, color = "red", linewidth = 1, linetype = "dashed") +
  labs(title = "LSTM Residual Distribution",
       subtitle = "Should be centered around 0",
       x = "Residual (Actual - Predicted)", y = "Frequency") +
  theme_minimal(base_size = 13)
ggsave("output/plots/lstm_residual_distribution.png", p_resid,
       width = 8, height = 6, dpi = 300)
cat("  Saved: output/plots/lstm_residual_distribution.png\n")

# Residuals over time
p_resid_time <- ggplot(result_df, aes(x = Index, y = Residual)) +
  geom_line(color = "gray40") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "LSTM Residuals Over Time",
       x = "Time Index", y = "Residual (min)") +
  theme_minimal(base_size = 13)
ggsave("output/plots/lstm_residuals_over_time.png", p_resid_time,
       width = 12, height = 5, dpi = 300)
cat("  Saved: output/plots/lstm_residuals_over_time.png\n")

cat("\n=== PHASE 8 COMPLETE ===\n")
