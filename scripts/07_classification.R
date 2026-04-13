# =============================================================================
# PHASE 7: MACHINE LEARNING — CLASSIFICATION
# Script: 07_classification.R
# Purpose: Train Logistic Regression, Random Forest, and SVM models to predict
#          delays. Evaluate with Confusion Matrix, ROC, AUC, F1. Compare models.
# =============================================================================

cat("=== PHASE 7: MACHINE LEARNING — CLASSIFICATION ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)
library(ggplot2)
library(gridExtra)

# --- Load transformed data ----------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
cat(sprintf("Loaded: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 7.1: Define Target Variable
# =============================================================================
cat("Step 7.1: Preparing target variable...\n")

df$delayed <- factor(df$delayed, levels = c("0", "1"), labels = c("No", "Yes"))
cat(sprintf("  Target (delayed) distribution:\n"))
print(table(df$delayed))
cat(sprintf("  Proportion: %.1f%% No, %.1f%% Yes\n",
            prop.table(table(df$delayed))[1] * 100,
            prop.table(table(df$delayed))[2] * 100))

# =============================================================================
# Step 7.2: Select Features for Classification
# =============================================================================
cat("\nStep 7.2: Selecting features...\n")

# Only pre-trip-known features (exclude actual delay columns)
feature_cols <- c(
  # Weather
  "temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
  "weather_condition_enc",
  # External events
  "event_type_enc", "event_attendance_est", "traffic_congestion_index",
  # Temporal
  "hour", "weekday", "is_weekend", "month",
  # Derived
  "weather_severity_index", "event_impact_score", "trip_duration_scheduled",
  "feels_like_impact", "congestion_weather_combo",
  # Transport
  "transport_type_enc",
  # Encoded season
  "season_enc"
)

# Filter to available columns
feature_cols <- feature_cols[feature_cols %in% names(df)]
cat(sprintf("  Using %d features: %s\n", length(feature_cols),
            paste(feature_cols, collapse = ", ")))

# Prepare feature matrix
model_df <- df[, c(feature_cols, "delayed")]
model_df <- model_df %>% mutate(across(-delayed, as.numeric))

# Handle binary columns that are factors
for (col in c("is_weekend")) {
  if (col %in% names(model_df) && is.factor(model_df[[col]])) {
    model_df[[col]] <- as.numeric(as.character(model_df[[col]]))
  }
}

# Remove any rows with NA in features
model_df <- model_df[complete.cases(model_df), ]
cat(sprintf("  Model dataset: %d rows x %d columns (inc. target)\n",
            nrow(model_df), ncol(model_df)))

# =============================================================================
# Step 7.3: Train-Test Split
# =============================================================================
cat("\nStep 7.3: Splitting data (80/20, stratified)...\n")

set.seed(42)
train_index <- createDataPartition(model_df$delayed, p = 0.8, list = FALSE)
train_data <- model_df[train_index, ]
test_data  <- model_df[-train_index, ]

cat(sprintf("  Training set: %d rows (%s)\n", nrow(train_data),
            paste(table(train_data$delayed), collapse = " No / ") ))
cat(sprintf("  Test set: %d rows (%s)\n", nrow(test_data),
            paste(table(test_data$delayed), collapse = " No / ")))

# =============================================================================
# Step 7.4: Handle Class Imbalance
# =============================================================================
cat("\nStep 7.4: Checking class imbalance...\n")

class_prop <- prop.table(table(train_data$delayed))
minority_pct <- min(class_prop) * 100

sampling_method <- "none"
if (minority_pct < 35) {
  sampling_method <- "up"
  cat(sprintf("  Class imbalance detected (minority: %.1f%%). Using UP-sampling.\n",
              minority_pct))
} else {
  cat(sprintf("  Classes are reasonably balanced (minority: %.1f%%). No resampling.\n",
              minority_pct))
}

# =============================================================================
# Step 7.5–7.7: Train Models with 5-Fold CV
# =============================================================================
cat("\nStep 7.5–7.7: Training models with 5-fold cross-validation...\n")

# Common trainControl
tc <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling        = if (sampling_method != "none") sampling_method else NULL
)

# --- Model 1: Logistic Regression (Baseline) ---------------------------------
cat("\n  Training Model 1: Logistic Regression...\n")
set.seed(42)
model_lr <- train(
  delayed ~ .,
  data   = train_data,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = tc
)
cat(sprintf("  LR training ROC: %.4f\n", max(model_lr$results$ROC)))

# --- Model 2: Random Forest --------------------------------------------------
cat("\n  Training Model 2: Random Forest...\n")
set.seed(42)
model_rf <- train(
  delayed ~ .,
  data   = train_data,
  method = "rf",
  ntree  = 100,
  metric = "ROC",
  trControl = tc
)
cat(sprintf("  RF training ROC: %.4f\n", max(model_rf$results$ROC)))

# --- Model 3: SVM (Radial Basis Function) ------------------------------------
cat("\n  Training Model 3: SVM (Radial)...\n")
set.seed(42)
model_svm <- train(
  delayed ~ .,
  data       = train_data,
  method     = "svmRadial",
  metric     = "ROC",
  tuneLength = 5,
  trControl  = tc
)
cat(sprintf("  SVM training ROC: %.4f\n", max(model_svm$results$ROC)))

# =============================================================================
# Step 7.8: Model Comparison on Test Set
# =============================================================================
cat("\nStep 7.8: Evaluating models on test set...\n")

# Predict on test set
pred_lr  <- predict(model_lr,  newdata = test_data)
pred_rf  <- predict(model_rf,  newdata = test_data)
pred_svm <- predict(model_svm, newdata = test_data)

prob_lr  <- predict(model_lr,  newdata = test_data, type = "prob")$Yes
prob_rf  <- predict(model_rf,  newdata = test_data, type = "prob")$Yes
prob_svm <- predict(model_svm, newdata = test_data, type = "prob")$Yes

# Confusion matrices
cm_lr  <- confusionMatrix(pred_lr,  test_data$delayed, positive = "Yes")
cm_rf  <- confusionMatrix(pred_rf,  test_data$delayed, positive = "Yes")
cm_svm <- confusionMatrix(pred_svm, test_data$delayed, positive = "Yes")

# AUC
auc_lr  <- auc(roc(test_data$delayed, prob_lr,  levels = c("No", "Yes")))
auc_rf  <- auc(roc(test_data$delayed, prob_rf,  levels = c("No", "Yes")))
auc_svm <- auc(roc(test_data$delayed, prob_svm, levels = c("No", "Yes")))

# Extract metrics
extract_metrics <- function(cm, auc_val, model_name) {
  data.frame(
    Model       = model_name,
    Accuracy    = round(cm$overall["Accuracy"], 4),
    Precision   = round(cm$byClass["Precision"], 4),
    Recall      = round(cm$byClass["Recall"], 4),
    F1_Score    = round(cm$byClass["F1"], 4),
    Specificity = round(cm$byClass["Specificity"], 4),
    AUC         = round(as.numeric(auc_val), 4),
    stringsAsFactors = FALSE
  )
}

comparison <- bind_rows(
  extract_metrics(cm_lr,  auc_lr,  "Logistic Regression"),
  extract_metrics(cm_rf,  auc_rf,  "Random Forest"),
  extract_metrics(cm_svm, auc_svm, "SVM (Radial)")
)

cat("\n  Model Comparison Table:\n")
print(comparison, row.names = FALSE)

write_csv(comparison, "output/reports/model_comparison.csv")
cat("  Saved: output/reports/model_comparison.csv\n")

# =============================================================================
# Visualizations
# =============================================================================
cat("\n  Creating visualizations...\n")

# ROC curves overlaid
roc_lr  <- roc(test_data$delayed, prob_lr,  levels = c("No", "Yes"))
roc_rf  <- roc(test_data$delayed, prob_rf,  levels = c("No", "Yes"))
roc_svm <- roc(test_data$delayed, prob_svm, levels = c("No", "Yes"))

png("output/plots/roc_curves_comparison.png", width = 1000, height = 800, res = 150)
plot(roc_lr, col = "blue", lwd = 2, main = "ROC Curves — Model Comparison")
plot(roc_rf, col = "green", lwd = 2, add = TRUE)
plot(roc_svm, col = "red", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         sprintf("Logistic Regression (AUC=%.3f)", auc_lr),
         sprintf("Random Forest (AUC=%.3f)", auc_rf),
         sprintf("SVM Radial (AUC=%.3f)", auc_svm)
       ),
       col = c("blue", "green", "red"), lwd = 2)
dev.off()
cat("  Saved: output/plots/roc_curves_comparison.png\n")

# Confusion matrix heatmaps
plot_cm <- function(cm, title, filename) {
  cm_table <- as.data.frame(cm$table)
  p <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 8, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 14)
  ggsave(paste0("output/plots/", filename), p, width = 6, height = 5, dpi = 300)
  cat(sprintf("  Saved: output/plots/%s\n", filename))
}

plot_cm(cm_lr,  "Confusion Matrix — Logistic Regression", "cm_logistic_regression.png")
plot_cm(cm_rf,  "Confusion Matrix — Random Forest",       "cm_random_forest.png")
plot_cm(cm_svm, "Confusion Matrix — SVM (Radial)",        "cm_svm_radial.png")

# Variable importance (Random Forest)
imp <- varImp(model_rf)
p_imp <- ggplot(imp) +
  labs(title = "Random Forest — Variable Importance") +
  theme_minimal(base_size = 13)
ggsave("output/plots/rf_variable_importance.png", p_imp, width = 10, height = 7, dpi = 300)
cat("  Saved: output/plots/rf_variable_importance.png\n")

# Model comparison grouped bar chart
comp_long <- comparison %>%
  tidyr::pivot_longer(cols = c(Accuracy, F1_Score, AUC),
                       names_to = "Metric", values_to = "Value")

p_comp <- ggplot(comp_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(Value, 3)), position = position_dodge(0.9),
            vjust = -0.3, size = 3.5) +
  labs(title = "Model Comparison — Accuracy, F1, AUC",
       x = "Metric", y = "Score") +
  theme_minimal(base_size = 13) +
  scale_fill_brewer(palette = "Set2") +
  coord_cartesian(ylim = c(0, 1))
ggsave("output/plots/model_comparison_bar.png", p_comp, width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/model_comparison_bar.png\n")

# =============================================================================
# Step 7.9: Select & Save Best Model
# =============================================================================
cat("\nStep 7.9: Selecting and saving the best model...\n")

# Pick best by F1 Score
best_idx <- which.max(comparison$F1_Score)
best_model_name <- comparison$Model[best_idx]
cat(sprintf("  Best model: %s (F1=%.4f, AUC=%.4f)\n",
            best_model_name, comparison$F1_Score[best_idx], comparison$AUC[best_idx]))

# Save the best model
best_model <- switch(best_model_name,
                     "Logistic Regression" = model_lr,
                     "Random Forest"       = model_rf,
                     "SVM (Radial)"        = model_svm)

saveRDS(best_model, "output/models/best_classification_model.rds")
cat("  Saved: output/models/best_classification_model.rds\n")

# Save all models
saveRDS(model_lr,  "output/models/model_logistic_regression.rds")
saveRDS(model_rf,  "output/models/model_random_forest.rds")
saveRDS(model_svm, "output/models/model_svm_radial.rds")
cat("  Saved all individual model files.\n")

# =============================================================================
# Step 7 (Bonus): 10-Fold Cross-Validation on Best Model
# =============================================================================
cat("\n  Running 10-fold cross-validation on best model...\n")

tc_10 <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling        = if (sampling_method != "none") sampling_method else NULL
)

set.seed(42)
cv10_model <- train(
  delayed ~ .,
  data      = model_df,
  method    = switch(best_model_name,
                     "Logistic Regression" = "glm",
                     "Random Forest"       = "rf",
                     "SVM (Radial)"        = "svmRadial"),
  metric    = "ROC",
  trControl = tc_10
)

cat("  10-Fold CV Results:\n")
cv_results <- cv10_model$results
if (best_model_name == "Logistic Regression") {
  cat(sprintf("    Mean ROC: %.4f, Mean Sens: %.4f, Mean Spec: %.4f\n",
              cv_results$ROC, cv_results$Sens, cv_results$Spec))
} else {
  best_row <- cv_results[which.max(cv_results$ROC), ]
  cat(sprintf("    Best ROC: %.4f, Sens: %.4f, Spec: %.4f\n",
              best_row$ROC, best_row$Sens, best_row$Spec))
}

# =============================================================================
# Error Analysis
# =============================================================================
cat("\n  Performing error analysis...\n")

# Identify false positives and false negatives
test_data$predicted <- pred_rf  # Use RF predictions regardless for error analysis
test_data$pred_prob <- prob_rf

test_data$error_type <- case_when(
  test_data$delayed == "Yes" & test_data$predicted == "No"  ~ "False Negative",
  test_data$delayed == "No"  & test_data$predicted == "Yes" ~ "False Positive",
  TRUE ~ "Correct"
)

error_counts <- table(test_data$error_type)
cat("  Error distribution:\n")
print(error_counts)

# Plot error analysis
p_error <- ggplot(test_data, aes(x = traffic_congestion_index,
                                  y = weather_severity_index,
                                  color = error_type)) +
  geom_point(alpha = 0.5, size = 2) +
  labs(title = "Error Analysis: Congestion vs Weather Severity",
       x = "Traffic Congestion Index", y = "Weather Severity Index",
       color = "Error Type") +
  theme_minimal(base_size = 13) +
  scale_color_manual(values = c("Correct" = "gray70",
                                 "False Positive" = "orange",
                                 "False Negative" = "red"))
ggsave("output/plots/error_analysis_scatter.png", p_error,
       width = 10, height = 7, dpi = 300)
cat("  Saved: output/plots/error_analysis_scatter.png\n")

cat("\n=== PHASE 7 COMPLETE ===\n")
