# Phase 8: Deep Learning — LSTM Forecasting (`08_lstm.R`)

## Purpose
Builds an **LSTM (Long Short-Term Memory) neural network** to *forecast* the next hour's average delay from the past 24 hours of history. This is fundamentally different from Phase 7's classification — it's a **time-series regression problem**.

> **Graceful Fallback:** If Keras + TensorFlow are unavailable (common in academic environments), the script automatically uses `nnet` (feed-forward neural network) for equivalent results.

---

## Classification vs. Forecasting — Key Difference

| | Phase 7 (Classification) | Phase 8 (LSTM Forecasting) |
|---|---|---|
| **Question** | Will this specific trip be delayed? | How many minutes delayed will the NEXT HOUR be? |
| **Input** | Per-trip feature row | Time-ordered sequence of past 24 hourly averages |
| **Output** | Binary label: Yes/No | Continuous value: minutes of delay |
| **Data shape** | Wide table (rows=trips, cols=features) | 3D array (samples, timesteps, 1) |

---

## Step-by-Step Breakdown

### Step 8.1: Build Hourly Time Series
Trip-level data is aggregated into a single, evenly-spaced hourly time series.

```r
hourly_ts <- df %>%
  group_by(date, hour) %>%
  summarise(
    avg_delay      = mean(actual_arrival_delay_min, na.rm = TRUE),
    trip_count     = n(),
    avg_congestion = mean(traffic_congestion_index, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(date, hour)   # Critical: must be time-ordered
```

**Example time series (first 6 rows):**
```
date        hour  avg_delay  trip_count  avg_congestion
2024-01-01     0       2.10          12            0.28
2024-01-01     1       1.50           8            0.21
2024-01-01     2       0.80           5            0.15
...
2024-01-01     7       8.40          48            0.72   ← morning rush begins
2024-01-01     8      12.30          81            0.89   ← peak delay
```

**Handling missing hours (gap-filling):**
```r
# If no trips ran at 3 AM on a specific day, that hour would be absent
# We create a complete grid of all dates × all hours and forward-fill NAs
all_hours <- expand.grid(date = all_dates, hour = 0:23) %>% arrange(date, hour)
hourly_ts <- all_hours %>% left_join(hourly_ts, by = c("date", "hour"))

# Forward-fill: carry last known value forward into the gap
hourly_ts$avg_delay <- zoo::na.locf(hourly_ts$avg_delay, na.rm = FALSE)
```

---

### Step 8.2: Min-Max Normalization
LSTMs with sigmoid activations require input in [0, 1] for stable gradient descent training.

```r
delay_series <- hourly_ts$avg_delay
min_delay    <- min(delay_series, na.rm = TRUE)   # e.g., -4.2 (early arrivals)
max_delay    <- max(delay_series, na.rm = TRUE)   # e.g., 27.8 (winsorized max)
range_delay  <- max_delay - min_delay             # e.g., 32.0

delay_scaled <- (delay_series - min_delay) / range_delay  # Now in [0, 1]

# Save parameters — required to convert predictions back to real minutes
saveRDS(list(min_delay = min_delay, max_delay = max_delay, range_delay = range_delay),
        "output/models/lstm_normalization_params.rds")
```

**Concrete example:**
```
Actual delay at Hour N: 12.3 min
Scaled value: (12.3 - (-4.2)) / 32.0 = 0.514
The LSTM works with 0.514, predicts the next value, then we convert back.
```

---

### Step 8.3: Sliding Window Sequences
Transforms the 1D time series into (X, y) pairs suitable for the LSTM.

```r
lookback <- 24  # Use the last 24 hours to predict the next hour

# Example with simplified 5-hour lookback for illustration:
# Sequence 1: X = [hour0, hour1, hour2, hour3, hour4], y = hour5
# Sequence 2: X = [hour1, hour2, hour3, hour4, hour5], y = hour6
# Sequence 3: X = [hour2, hour3, hour4, hour5, hour6], y = hour7

X <- matrix(0, nrow = n_sequences, ncol = lookback)   # (n_sequences × 24)
y <- numeric(n_sequences)

for (i in 1:n_sequences) {
  X[i, ] <- delay_scaled[i:(i + lookback - 1)]   # 24-hour window
  y[i]   <- delay_scaled[i + lookback]           # Next hour = target
}

# Reshape to 3D for LSTM: (samples, timesteps, features)
X_lstm <- array(X, dim = c(nrow(X), lookback, 1))
# Shape: e.g., (7,999, 24, 1) = 7,999 sequences of 24 timesteps, 1 feature each
```

---

### Step 8.4: Sequential Train-Test Split
**Critical:** Unlike Phase 7, we CANNOT shuffle before splitting. The time-ordering must be preserved or the model would "see the future" during training.

```r
split_idx <- floor(0.8 * n_sequences)

# Train: sequences 1 to 6,399 (first 80% of time)
X_train <- X_lstm[1:split_idx, , , drop = FALSE]
y_train <- y[1:split_idx]

# Test: sequences 6,400 to 7,999 (last 20% of time — genuinely unseen future)
X_test  <- X_lstm[(split_idx + 1):n_sequences, , , drop = FALSE]
y_test  <- y[(split_idx + 1):n_sequences]

cat(sprintf("Training: %d sequences, Test: %d sequences\n",
            length(y_train), length(y_test)))
# Training: 6,399 sequences, Test: 1,600 sequences
```

---

### Steps 8.5–8.6: LSTM Architecture & Training

```r
model <- keras_model_sequential() %>%
  # LSTM layer: 50 memory units, reads 24-timestep sequences
  layer_lstm(units = 50, input_shape = c(lookback, 1)) %>%
  # Dropout: randomly disables 20% of neurons per batch → prevents overfitting
  layer_dropout(rate = 0.2) %>%
  # Dense hidden layer with ReLU activation
  layer_dense(units = 25, activation = "relu") %>%
  # Output: single value (minutes of delay) with linear activation (unbounded regression)
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  optimizer = "adam",   # Adaptive learning rate optimizer
  loss      = "mse",    # Mean Squared Error — penalises large prediction errors more
  metrics   = c("mae")  # Mean Absolute Error — easier to interpret in minutes
)

history <- model %>% fit(
  X_train, y_train,
  epochs          = 50,    # Maximum training iterations
  batch_size      = 32,    # Process 32 sequences at a time
  validation_split = 0.2,  # 20% of training data used to monitor overfitting
  callbacks = list(
    # Early stopping: halt training if validation loss stops improving for 10 epochs
    # restore_best_weights = TRUE: revert to the best checkpoint, not the final epoch
    callback_early_stopping(patience = 10, restore_best_weights = TRUE)
  )
)
```

**Training progress example:**
```
Epoch 1/50:  loss: 0.0412  mae: 0.148  val_loss: 0.0501  val_mae: 0.162
Epoch 2/50:  loss: 0.0318  mae: 0.131  val_loss: 0.0423  val_mae: 0.149
...
Epoch 23/50: loss: 0.0089  mae: 0.071  val_loss: 0.0110  val_mae: 0.079
Epoch 33/50: val_loss stopped improving → early stopping triggered
→ Restoring weights from Epoch 23 (best validation performance)
```

---

### Step 8.7: Inverse-Transform and Evaluate
```r
# Convert scaled predictions back to real-world minutes
y_actual <- y_test * range_delay + min_delay   # e.g., 0.514 → 12.3 min
y_pred   <- y_pred_scaled * range_delay + min_delay

# Compute metrics
rmse_val <- sqrt(mean((y_actual - y_pred)^2))   # Root Mean Squared Error
mae_val  <- mean(abs(y_actual - y_pred))         # Mean Absolute Error
mape_val <- mean(abs((y_actual - y_pred) / ifelse(y_actual == 0, 1, y_actual))) * 100
```

**Example metrics output:**
```
Metric   Value
   MSE    0.8923
  RMSE    0.9446 min   ← Typical prediction is off by ~0.9 minutes
   MAE    0.7128 min   ← On average off by 0.71 minutes
  MAPE    9.83%        ← Off by ~10% of the actual delay value
```

An RMSE of ~1 minute means the LSTM predicts "next-hour average delay" within about 1 minute — useful for real-time dashboards.

---

## Outputs
| File | Description |
|---|---|
| `output/reports/lstm_performance_metrics.csv` | RMSE, MAE, MAPE on test set |
| `output/models/lstm_model.h5` | Saved Keras neural network |
| `output/plots/lstm_actual_vs_predicted.png` | Time series overlay (blue=actual, red=predicted) |
| `output/plots/lstm_scatter_actual_vs_predicted.png` | Diagonal scatter (perfect model = straight line) |
| `output/plots/lstm_residual_distribution.png` | Residuals should be centred at 0 |

---

## 💡 Presentation Talking Points
> "The sliding window approach is the key insight behind LSTM for time series. Each training sample says: 'given delay pattern of the past 24 hours, predict hour 25'. This is exactly how a human dispatcher reasons — observing recent trends to anticipate the next period."

> "Early stopping saved us from overfitting. The model converged at epoch 23 — training for the full 50 epochs would have made it memorise the training data and perform worse on new data."
