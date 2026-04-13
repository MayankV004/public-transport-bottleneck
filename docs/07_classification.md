# Phase 7: Machine Learning — Classification (`07_classification.R`)

## Purpose
Trains, compares, and selects the best classifier to predict whether a trip will be delayed *before it departs* — using only features a dispatcher would know in advance. Three model families are compared to ensure the best one is chosen objectively.

---

## Step-by-Step Breakdown

### Step 7.1: Prepare Target Variable
The target `delayed` is converted from `"0"/"1"` strings to a proper factor with meaningful labels.

```r
df$delayed <- factor(df$delayed, levels = c("0", "1"), labels = c("No", "Yes"))
print(table(df$delayed))
# No: 6523   Yes: 3477
# Proportion: 65.2% No | 34.8% Yes → reasonably balanced
```

---

### Step 7.2: Feature Selection — No Leakage
Only features knowable *before* the trip departs are included. Actual delay columns are excluded.

```r
feature_cols <- c(
  # Pre-trip weather forecast
  "temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
  "weather_condition_enc",
  # Event & traffic context
  "event_type_enc", "event_attendance_est", "traffic_congestion_index",
  # Temporal
  "hour", "weekday", "is_weekend", "month",
  # Engineered interaction features (from Phase 3)
  "weather_severity_index", "event_impact_score", "trip_duration_scheduled",
  "feels_like_impact", "congestion_weather_combo",
  # Transport meta
  "transport_type_enc", "season_enc"
)
```

**Why exclude `actual_arrival_delay_min`?**
The model's job is to *predict* delays. Using the actual delay as a predictor would mean the model already knows the answer — this is called **target leakage** and produces falsely high accuracy.

**Example of leakage vs valid features:**

| Feature | Available Before Trip? | Include? |
|---|---|---|
| `weather_forecast` (temperature, rain) | ✅ Yes | ✅ Include |
| `traffic_congestion_index` | ✅ Yes | ✅ Include |
| `actual_arrival_delay_min` | ❌ No (unknown until trip ends) | ❌ Exclude |
| `delay_change` | ❌ No (compares dep & arr delay) | ❌ Exclude |

---

### Step 7.3: Stratified Train-Test Split
`createDataPartition` ensures both train and test have the same class ratio as the full dataset.

```r
set.seed(42)
train_index <- createDataPartition(model_df$delayed, p = 0.8, list = FALSE)
train_data  <- model_df[train_index, ]   # 80% = 7,980 rows
test_data   <- model_df[-train_index, ]  # 20% = 1,995 rows

cat(sprintf("Training set: %d rows\n", nrow(train_data)))
cat(sprintf("Test set: %d rows\n",     nrow(test_data)))
```

**Example output:**
```
Training set: 7,980 rows (65.2% No / 34.8% Yes)
Test set:     1,995 rows (65.2% No / 34.8% Yes)
```
The class ratio is preserved in both sets thanks to stratification.

---

### Step 7.4: Automatic Class Imbalance Handling
Dynamically detects imbalance and applies up-sampling if the minority class is under-represented.

```r
class_prop   <- prop.table(table(train_data$delayed))
minority_pct <- min(class_prop) * 100

if (minority_pct < 35) {
  sampling_method <- "up"
  cat(sprintf("Imbalance detected (minority: %.1f%%). Applying UP-sampling.\n", minority_pct))
} else {
  sampling_method <- "none"
  cat(sprintf("Classes balanced (minority: %.1f%%). No resampling needed.\n", minority_pct))
}
# Example output: "Classes balanced (minority: 34.8%). No resampling needed."
```

---

### Steps 7.5–7.7: Training with 5-Fold Cross-Validation
All three models share the same `trainControl` — a fair comparison "playing field".

```r
tc <- trainControl(
  method          = "cv",    # k-fold cross-validation
  number          = 5,       # 5 folds
  classProbs      = TRUE,    # Generate probabilities (needed for AUC/ROC)
  summaryFunction = twoClassSummary,  # Reports ROC, Sens, Spec
  savePredictions = "final"  # Keep fold predictions for analysis
)
```

**What 5-fold cross-validation means:**
1. Split training data into 5 equal parts
2. Train on 4 parts, evaluate on the 5th
3. Repeat 5 times (each part gets to be the validation set once)
4. Average the 5 AUC scores → more reliable estimate than a single train/test split

```r
# Model 1: Logistic Regression (linear baseline)
model_lr <- train(delayed ~ ., data = train_data,
                  method = "glm", family = "binomial",
                  metric = "ROC", trControl = tc)
# "delayed ~ ." means: predict 'delayed' from all other columns

# Model 2: Random Forest (ensemble of 100 decision trees)
model_rf <- train(delayed ~ ., data = train_data,
                  method = "rf", ntree = 100,
                  metric = "ROC", trControl = tc)

# Model 3: SVM with Radial Basis Function kernel
model_svm <- train(delayed ~ ., data = train_data,
                   method = "svmRadial",
                   metric = "ROC", tuneLength = 5,  # Test 5 combinations of C and sigma
                   trControl = tc)
```

---

### Step 7.8: Model Comparison on Test Set

```r
# Get predicted class labels and probabilities from all 3 models
pred_lr  <- predict(model_lr,  newdata = test_data)           # "No"/"Yes"
prob_lr  <- predict(model_lr,  newdata = test_data, type = "prob")$Yes  # Probability of Yes

# Confusion matrix: shows TP, TN, FP, FN
cm_rf <- confusionMatrix(pred_rf, test_data$delayed, positive = "Yes")
```

**Example confusion matrix for Random Forest:**
```
                Reference
Prediction    No    Yes
       No    1242    98   ← True Negatives (1242) + False Negatives (98)
      Yes      57   598   ← False Positives (57) + True Positives (598)
```

**Derived metrics:**
```
Accuracy    = (1242 + 598) / 1995 = 92.2%
Precision   = 598 / (57 + 598)   = 91.3%  (when we predict delayed, how often right?)
Recall      = 598 / (598 + 98)   = 85.9%  (of all actual delays, how many caught?)
F1 Score    = 2 × (Precision × Recall) / (Precision + Recall) = 88.5%
AUC         = 0.947  (area under ROC curve — closer to 1.0 is perfect)
```

**Why F1 Score for model selection?**
F1 balances precision and recall. In delay prediction:
- **False Negative (missed delay):** Passengers miss connections → high operational cost
- **False Positive (false alarm):** Unnecessary intervention dispatched → minor cost
F1 penalises both fairly, making it the right metric here.

---

### Error Analysis
```r
test_data$error_type <- case_when(
  test_data$delayed == "Yes" & test_data$predicted == "No"  ~ "False Negative",
  test_data$delayed == "No"  & test_data$predicted == "Yes" ~ "False Positive",
  TRUE                                                      ~ "Correct"
)

# Plot errors in feature space (congestion × weather severity)
# Areas with many red dots (False Negatives) = model blind spots
```

**Reading the error plot:**
- **Gray dots (Correct):** Model confident here
- **Red dots (False Negative):** Missed real delays → likely at medium congestion/medium weather where the boundary is unclear
- **Orange dots (False Positive):** False alarms → likely cases where conditions looked bad but the trip happened to run on time

---

## Outputs
| File | Description |
|---|---|
| `output/reports/model_comparison.csv` | Accuracy, Precision, Recall, F1, AUC for all 3 models |
| `output/models/best_classification_model.rds` | Best model by F1 Score |
| `output/plots/roc_curves_comparison.png` | Overlaid ROC curves |
| `output/plots/cm_*.png` | Confusion matrix heatmaps (one per model) |
| `output/plots/rf_variable_importance.png` | Feature importance ranking |
| `output/plots/error_analysis_scatter.png` | Where models fail in feature space |

---

## 💡 Presentation Talking Points
> "We trained three fundamentally different model architectures — linear (LR), tree ensemble (RF), and kernel-based (SVM). Because they make different assumptions, their agreement tells us the prediction is robust. If all three agree a trip will be delayed, we can be very confident."

> "Random Forest's variable importance plot reveals that `congestion_weather_combo` — the interaction feature we engineered in Phase 3 — is a top predictor. This proves that our domain knowledge improved model performance."
