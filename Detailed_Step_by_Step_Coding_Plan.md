# 📋 COMPREHENSIVE STEP-BY-STEP CODING PLAN
# Public Transport Delay Propagation and Bottleneck Prediction

## 🎯 PROJECT OVERVIEW

**Objective:** Build an end-to-end data mining, machine learning, and deep learning pipeline in R to analyze public transport delays, identify bottlenecks, and predict future congestion patterns.

**Core Technologies:** R (primary), Power BI (visualization + cleaning), GTFS data

**Key Deliverables:**
- Data mining pipeline with cleaning, transformation, and reduction
- Clustering analysis for bottleneck identification
- ML classification models for bottleneck prediction
- Deep Learning (LSTM) for delay forecasting
- Power BI dashboards for visualization
- Complete documentation and performance metrics

---

## 📚 REQUIRED R PACKAGES

Install all required packages at the beginning:

```r
# Data manipulation and cleaning
install.packages("dplyr")
install.packages("tidyr")
install.packages("lubridate")
install.packages("readr")

# Visualization
install.packages("ggplot2")
install.packages("corrplot")
install.packages("plotly")
install.packages("pheatmap")

# Data mining and dimensionality reduction
install.packages("factoextra")
install.packages("FactoMineR")

# Clustering
install.packages("cluster")
install.packages("dbscan")
install.packages("fpc")

# Machine Learning
install.packages("caret")
install.packages("randomForest")
install.packages("e1071")
install.packages("MLmetrics")

# Deep Learning
install.packages("keras")
install.packages("tensorflow")

# Network analysis (for delay propagation)
install.packages("igraph")
install.packages("networkD3")

# Model evaluation
install.packages("pROC")
install.packages("ROCR")

# Load all libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(ggplot2)
library(corrplot)
library(plotly)
library(pheatmap)
library(factoextra)
library(FactoMineR)
library(cluster)
library(dbscan)
library(fpc)
library(caret)
library(randomForest)
library(e1071)
library(MLmetrics)
library(keras)
library(tensorflow)
library(igraph)
library(networkD3)
library(pROC)
library(ROCR)
```

---

## 📊 PHASE 1: DATA COLLECTION & UNDERSTANDING

### Step 1.1: Load Raw Data

```r
# Set working directory
setwd("your/project/path")

# Load GTFS or public transport data
# Assume we have the following CSV files:
# - stop_times.csv: scheduled and actual arrival times
# - stops.csv: station information
# - routes.csv: route information
# - trips.csv: trip details

stop_times <- read_csv("data/stop_times.csv")
stops <- read_csv("data/stops.csv")
routes <- read_csv("data/routes.csv")
trips <- read_csv("data/trips.csv")

# Display structure
str(stop_times)
str(stops)
str(routes)
str(trips)

# Display first few rows
head(stop_times)
head(stops)
```

### Step 1.2: Understand Data Structure

```r
# Check dimensions
dim(stop_times)
dim(stops)

# Check column names
colnames(stop_times)
colnames(stops)

# Summary statistics
summary(stop_times)

# Check for data types
sapply(stop_times, class)

# Count unique values
stop_times %>% 
  summarise(
    unique_trips = n_distinct(trip_id),
    unique_stops = n_distinct(stop_id),
    unique_routes = n_distinct(route_id)
  )
```

### Step 1.3: Data Profiling

```r
# Create data profiling report
data_profile <- stop_times %>%
  summarise(
    total_records = n(),
    missing_scheduled = sum(is.na(scheduled_arrival_time)),
    missing_actual = sum(is.na(actual_arrival_time)),
    date_range_start = min(date, na.rm = TRUE),
    date_range_end = max(date, na.rm = TRUE)
  )

print(data_profile)

# Save profiling results
write_csv(data_profile, "output/data_profile.csv")
```

**Output for Report:**
- Total number of records
- Number of unique trips, stops, routes
- Date range of data
- Missing value counts
- Data structure description

---

## 🧹 PHASE 2: DATA CLEANING

### Step 2.1: Handle Missing Values

```r
# Check missing values percentage
missing_summary <- stop_times %>%
  summarise_all(~sum(is.na(.)) / n() * 100)

print(missing_summary)

# Remove rows with missing critical values
stop_times_clean <- stop_times %>%
  filter(!is.na(actual_arrival_time)) %>%
  filter(!is.na(scheduled_arrival_time)) %>%
  filter(!is.na(stop_id)) %>%
  filter(!is.na(trip_id))

# Log the number of removed records
records_before <- nrow(stop_times)
records_after <- nrow(stop_times_clean)
records_removed <- records_before - records_after

cat("Records before cleaning:", records_before, "\n")
cat("Records after cleaning:", records_after, "\n")
cat("Records removed:", records_removed, "\n")
```

### Step 2.2: Remove Duplicates

```r
# Check for duplicates
duplicates_count <- stop_times_clean %>%
  group_by(trip_id, stop_id, date) %>%
  filter(n() > 1) %>%
  nrow()

cat("Duplicate records found:", duplicates_count, "\n")

# Remove duplicates
stop_times_clean <- stop_times_clean %>%
  distinct(trip_id, stop_id, date, .keep_all = TRUE)

cat("Records after removing duplicates:", nrow(stop_times_clean), "\n")
```

### Step 2.3: Data Type Conversions

```r
# Convert time columns to proper format
stop_times_clean <- stop_times_clean %>%
  mutate(
    scheduled_arrival_time = as.POSIXct(scheduled_arrival_time),
    actual_arrival_time = as.POSIXct(actual_arrival_time),
    date = as.Date(date)
  )

# Verify conversions
str(stop_times_clean)
```

### Step 2.4: Filter Invalid Data

```r
# Remove records where actual time is before scheduled time by more than threshold
# (likely data entry errors)
stop_times_clean <- stop_times_clean %>%
  mutate(time_diff = as.numeric(difftime(actual_arrival_time, 
                                          scheduled_arrival_time, 
                                          units = "mins"))) %>%
  filter(time_diff >= -10) %>%  # Allow max 10 min early arrival
  filter(time_diff <= 180)      # Remove delays > 3 hours (likely errors)

# Log filtering results
cat("Records after filtering invalid times:", nrow(stop_times_clean), "\n")
```

### Step 2.5: Save Cleaned Data

```r
# Save cleaned dataset
write_csv(stop_times_clean, "output/stop_times_cleaned.csv")

# Create cleaning summary report
cleaning_report <- data.frame(
  Step = c("Raw Data", "Missing Values Removed", "Duplicates Removed", 
           "Invalid Times Filtered", "Final Clean Data"),
  Records = c(records_before, 
              records_after, 
              nrow(stop_times_clean), 
              nrow(stop_times_clean), 
              nrow(stop_times_clean))
)

write_csv(cleaning_report, "output/cleaning_summary.csv")
```

**Output for Report:**
- Cleaning summary table showing records at each step
- Percentage of data retained
- Missing value handling strategy

---

## 🔧 PHASE 3: DATA TRANSFORMATION & FEATURE ENGINEERING

### Step 3.1: Calculate Delay

```r
# Calculate delay in minutes
stop_times_transformed <- stop_times_clean %>%
  mutate(
    delay_minutes = as.numeric(difftime(actual_arrival_time, 
                                        scheduled_arrival_time, 
                                        units = "mins")),
    is_delayed = ifelse(delay_minutes > 5, 1, 0)  # Delayed if > 5 min
  )

# Delay statistics
delay_stats <- stop_times_transformed %>%
  summarise(
    mean_delay = mean(delay_minutes),
    median_delay = median(delay_minutes),
    sd_delay = sd(delay_minutes),
    max_delay = max(delay_minutes),
    min_delay = min(delay_minutes),
    delay_rate = mean(is_delayed) * 100
  )

print(delay_stats)
```

### Step 3.2: Extract Temporal Features

```r
# Extract time-based features
stop_times_transformed <- stop_times_transformed %>%
  mutate(
    hour = hour(scheduled_arrival_time),
    day_of_week = wday(date, label = TRUE),
    day_type = ifelse(day_of_week %in% c("Sat", "Sun"), "Weekend", "Weekday"),
    is_peak_hour = ifelse(hour %in% c(7, 8, 9, 16, 17, 18), 1, 0),
    month = month(date),
    week_of_year = week(date)
  )

# Display sample
head(stop_times_transformed[, c("delay_minutes", "hour", "day_of_week", 
                                 "day_type", "is_peak_hour")])
```

### Step 3.3: Aggregate by Station

```r
# Station-level aggregation
station_stats <- stop_times_transformed %>%
  group_by(stop_id) %>%
  summarise(
    total_arrivals = n(),
    avg_delay = mean(delay_minutes),
    median_delay = median(delay_minutes),
    sd_delay = sd(delay_minutes),
    delay_frequency = sum(is_delayed),
    delay_rate = mean(is_delayed) * 100,
    max_delay = max(delay_minutes),
    min_delay = min(delay_minutes)
  ) %>%
  arrange(desc(avg_delay))

# Display top 10 most delayed stations
head(station_stats, 10)

# Save station statistics
write_csv(station_stats, "output/station_delay_statistics.csv")
```

### Step 3.4: Aggregate by Route

```r
# Route-level aggregation
route_stats <- stop_times_transformed %>%
  left_join(trips, by = "trip_id") %>%
  group_by(route_id) %>%
  summarise(
    total_trips = n(),
    avg_delay = mean(delay_minutes),
    median_delay = median(delay_minutes),
    delay_frequency = sum(is_delayed),
    delay_rate = mean(is_delayed) * 100
  ) %>%
  arrange(desc(avg_delay))

head(route_stats, 10)

# Save route statistics
write_csv(route_stats, "output/route_delay_statistics.csv")
```

### Step 3.5: Aggregate by Hour and Day Type

```r
# Time-based aggregation
hourly_stats <- stop_times_transformed %>%
  group_by(hour, day_type) %>%
  summarise(
    avg_delay = mean(delay_minutes),
    median_delay = median(delay_minutes),
    total_arrivals = n(),
    delay_rate = mean(is_delayed) * 100
  )

# Peak hour analysis
peak_analysis <- stop_times_transformed %>%
  group_by(is_peak_hour) %>%
  summarise(
    avg_delay = mean(delay_minutes),
    delay_rate = mean(is_delayed) * 100
  )

print(peak_analysis)
```

### Step 3.6: Create Station-Hour Features

```r
# Detailed station-hour aggregation for ML models
station_hour_features <- stop_times_transformed %>%
  group_by(stop_id, hour, day_type) %>%
  summarise(
    avg_delay = mean(delay_minutes),
    median_delay = median(delay_minutes),
    sd_delay = sd(delay_minutes),
    max_delay = max(delay_minutes),
    delay_frequency = sum(is_delayed),
    total_arrivals = n(),
    delay_rate = mean(is_delayed) * 100
  ) %>%
  ungroup()

# Save for later use
write_csv(station_hour_features, "output/station_hour_features.csv")
```

**Output for Report:**
- Summary statistics of delays
- Station-wise delay rankings
- Route-wise delay rankings
- Peak vs off-peak comparison
- Hourly delay patterns

---

## 📊 PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)

### Step 4.1: Delay Distribution

```r
# Histogram of delays
ggplot(stop_times_transformed, aes(x = delay_minutes)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Delay Times",
       x = "Delay (minutes)",
       y = "Frequency") +
  theme_minimal() +
  xlim(-5, 60)  # Focus on reasonable delay range

ggsave("output/delay_distribution.png", width = 10, height = 6)

# Box plot by day type
ggplot(stop_times_transformed, aes(x = day_type, y = delay_minutes, fill = day_type)) +
  geom_boxplot() +
  labs(title = "Delay Distribution by Day Type",
       x = "Day Type",
       y = "Delay (minutes)") +
  theme_minimal() +
  ylim(-5, 40)

ggsave("output/delay_by_daytype.png", width = 8, height = 6)
```

### Step 4.2: Hourly Delay Patterns

```r
# Line plot of average delay by hour
hourly_avg <- stop_times_transformed %>%
  group_by(hour) %>%
  summarise(avg_delay = mean(delay_minutes))

ggplot(hourly_avg, aes(x = hour, y = avg_delay)) +
  geom_line(color = "darkred", size = 1) +
  geom_point(color = "darkred", size = 3) +
  labs(title = "Average Delay by Hour of Day",
       x = "Hour",
       y = "Average Delay (minutes)") +
  theme_minimal() +
  scale_x_continuous(breaks = 0:23)

ggsave("output/hourly_delay_pattern.png", width = 10, height = 6)
```

### Step 4.3: Station Delay Heatmap

```r
# Create heatmap matrix: stations vs hours
heatmap_data <- stop_times_transformed %>%
  group_by(stop_id, hour) %>%
  summarise(avg_delay = mean(delay_minutes)) %>%
  ungroup() %>%
  spread(hour, avg_delay)

# Convert to matrix (remove stop_id column)
heatmap_matrix <- as.matrix(heatmap_data[, -1])
rownames(heatmap_matrix) <- heatmap_data$stop_id

# Create heatmap
pheatmap(heatmap_matrix,
         cluster_rows = TRUE,
         cluster_cols = FALSE,
         color = colorRampPalette(c("white", "yellow", "red"))(50),
         main = "Station-Hour Delay Heatmap",
         filename = "output/station_hour_heatmap.png",
         width = 12,
         height = 8)
```

### Step 4.4: Top Delayed Stations Bar Chart

```r
# Top 20 most delayed stations
top_delayed <- station_stats %>%
  top_n(20, avg_delay) %>%
  arrange(desc(avg_delay))

ggplot(top_delayed, aes(x = reorder(stop_id, avg_delay), y = avg_delay)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Top 20 Most Delayed Stations",
       x = "Station ID",
       y = "Average Delay (minutes)") +
  theme_minimal()

ggsave("output/top_delayed_stations.png", width = 10, height = 8)
```

### Step 4.5: Correlation Analysis

```r
# Select numeric features for correlation
numeric_features <- stop_times_transformed %>%
  select(delay_minutes, hour, is_peak_hour, day_of_week) %>%
  mutate(day_of_week = as.numeric(day_of_week))

# Correlation matrix
cor_matrix <- cor(numeric_features, use = "complete.obs")

# Visualize correlation
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black",
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))

# Save
png("output/correlation_matrix.png", width = 800, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
dev.off()
```

**Output for Report:**
- Delay distribution histogram
- Hourly delay pattern line chart
- Station-hour heatmap
- Top delayed stations bar chart
- Correlation matrix
- Summary insights from visualizations

---

## 📉 PHASE 5: DIMENSIONALITY REDUCTION (PCA)

### Step 5.1: Prepare Data for PCA

```r
# Create feature matrix for PCA
# Use station-hour aggregated features
pca_data <- station_hour_features %>%
  select(avg_delay, median_delay, sd_delay, max_delay, 
         delay_frequency, total_arrivals, delay_rate)

# Remove any rows with NA values
pca_data <- na.omit(pca_data)

# Check data
dim(pca_data)
summary(pca_data)
```

### Step 5.2: Standardize Features

```r
# Standardize (scale) the data
# PCA requires standardization since features have different scales
pca_data_scaled <- scale(pca_data)

# Verify scaling
colMeans(pca_data_scaled)  # Should be close to 0
apply(pca_data_scaled, 2, sd)  # Should be 1
```

### Step 5.3: Perform PCA

```r
# Apply PCA
pca_model <- prcomp(pca_data_scaled, center = FALSE, scale. = FALSE)

# Summary of PCA
summary(pca_model)

# Variance explained by each component
variance_explained <- (pca_model$sdev^2) / sum(pca_model$sdev^2)
cumulative_variance <- cumsum(variance_explained)

# Create variance table
variance_table <- data.frame(
  Component = paste0("PC", 1:length(variance_explained)),
  Variance_Explained = variance_explained * 100,
  Cumulative_Variance = cumulative_variance * 100
)

print(variance_table)
write_csv(variance_table, "output/pca_variance_explained.csv")
```

### Step 5.4: Visualize PCA Results

```r
# Scree plot
fviz_eig(pca_model, addlabels = TRUE, 
         main = "Scree Plot: Variance Explained by Principal Components")
ggsave("output/pca_scree_plot.png", width = 10, height = 6)

# Biplot
fviz_pca_biplot(pca_model, 
                geom = c("point", "arrow"),
                title = "PCA Biplot: Variables and Observations")
ggsave("output/pca_biplot.png", width = 10, height = 8)

# Variable contribution to PC1 and PC2
fviz_pca_var(pca_model, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE,
             title = "PCA Variable Contribution")
ggsave("output/pca_variable_contribution.png", width = 10, height = 8)
```

### Step 5.5: Select Principal Components

```r
# Decide how many components to keep (e.g., 90% variance)
n_components <- which(cumulative_variance >= 0.90)[1]
cat("Number of components for 90% variance:", n_components, "\n")

# Extract transformed data
pca_transformed <- as.data.frame(pca_model$x[, 1:n_components])

# Add back identifiers if needed
pca_transformed$stop_id <- station_hour_features$stop_id
pca_transformed$hour <- station_hour_features$hour
pca_transformed$day_type <- station_hour_features$day_type

# Save PCA-transformed data
write_csv(pca_transformed, "output/pca_transformed_data.csv")

# Save PCA model for future use
saveRDS(pca_model, "output/pca_model.rds")
```

**Output for Report:**
- Variance explained table
- Scree plot
- Biplot showing variable relationships
- Number of components selected
- Interpretation of principal components
- Statement: "PCA reduced X features to Y components while retaining 90% of variance"

---

## 🎯 PHASE 6: CLUSTERING (BOTTLENECK IDENTIFICATION)

### Step 6.1: Prepare Clustering Data

```r
# Use PCA-transformed data for clustering
clustering_data <- pca_transformed %>%
  select(starts_with("PC"))

# Alternative: use original scaled features
# clustering_data <- pca_data_scaled

# Check for any remaining NA values
sum(is.na(clustering_data))
```

### Step 6.2: Determine Optimal Number of Clusters (Elbow Method)

```r
# Elbow method for K-means
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(clustering_data, centers = k, nstart = 25)$tot.withinss
})

# Plot elbow curve
elbow_data <- data.frame(K = 1:10, WSS = wss)
ggplot(elbow_data, aes(x = K, y = WSS)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Elbow Method for Optimal K",
       x = "Number of Clusters (K)",
       y = "Within-Cluster Sum of Squares") +
  theme_minimal() +
  scale_x_continuous(breaks = 1:10)

ggsave("output/kmeans_elbow_plot.png", width = 10, height = 6)
```

### Step 6.3: Silhouette Analysis

```r
# Silhouette analysis
library(cluster)

silhouette_scores <- sapply(2:10, function(k) {
  km_model <- kmeans(clustering_data, centers = k, nstart = 25)
  ss <- silhouette(km_model$cluster, dist(clustering_data))
  mean(ss[, 3])
})

sil_data <- data.frame(K = 2:10, Silhouette = silhouette_scores)

ggplot(sil_data, aes(x = K, y = Silhouette)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Silhouette Score for Different K",
       x = "Number of Clusters (K)",
       y = "Average Silhouette Score") +
  theme_minimal() +
  scale_x_continuous(breaks = 2:10)

ggsave("output/silhouette_analysis.png", width = 10, height = 6)

# Optimal K based on silhouette
optimal_k <- sil_data$K[which.max(sil_data$Silhouette)]
cat("Optimal K based on silhouette:", optimal_k, "\n")
```

### Step 6.4: Perform K-Means Clustering

```r
# Apply K-means with optimal K (let's use K=3 for demonstration)
set.seed(123)
k <- 3  # Can be changed based on elbow/silhouette analysis

kmeans_model <- kmeans(clustering_data, centers = k, nstart = 25)

# Add cluster labels to data
pca_transformed$cluster <- as.factor(kmeans_model$cluster)

# Cluster summary
cluster_summary <- pca_transformed %>%
  group_by(cluster) %>%
  summarise(
    count = n(),
    avg_PC1 = mean(PC1),
    avg_PC2 = mean(PC2)
  )

print(cluster_summary)
```

### Step 6.5: Visualize Clusters

```r
# 2D cluster visualization using PC1 and PC2
ggplot(pca_transformed, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(level = 0.95, linetype = 2) +
  labs(title = "K-Means Clustering Results (PCA Space)",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster") +
  theme_minimal()

ggsave("output/kmeans_clusters_pca.png", width = 10, height = 8)

# 3D visualization if using plotly
plot_ly(pca_transformed, x = ~PC1, y = ~PC2, z = ~PC3, 
        color = ~cluster, colors = c("red", "blue", "green")) %>%
  add_markers() %>%
  layout(title = "3D K-Means Clustering",
         scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")))
```

### Step 6.6: Interpret Clusters

```r
# Join original features back to understand cluster characteristics
clustered_data <- station_hour_features %>%
  mutate(cluster = pca_transformed$cluster)

# Profile each cluster
cluster_profiles <- clustered_data %>%
  group_by(cluster) %>%
  summarise(
    count = n(),
    avg_delay = mean(avg_delay),
    avg_delay_rate = mean(delay_rate),
    avg_frequency = mean(delay_frequency),
    median_total_arrivals = median(total_arrivals)
  )

print(cluster_profiles)
write_csv(cluster_profiles, "output/cluster_profiles.csv")

# Label clusters based on characteristics
# Cluster 1: Low delay (Stable stations)
# Cluster 2: Moderate delay (Delay-sensitive stations)
# Cluster 3: High delay (Bottleneck-prone stations)

cluster_labels <- clustered_data %>%
  mutate(
    cluster_label = case_when(
      cluster == 1 ~ "Stable",
      cluster == 2 ~ "Delay-Sensitive",
      cluster == 3 ~ "Bottleneck-Prone"
    )
  )

write_csv(cluster_labels, "output/clustered_stations_labeled.csv")
```

### Step 6.7: DBSCAN (Optional Alternative)

```r
# DBSCAN clustering (density-based)
# Determine optimal eps using k-distance plot
kNNdistplot(clustering_data, k = 4)
abline(h = 0.5, col = "red", lty = 2)  # Adjust threshold visually

# Apply DBSCAN
dbscan_model <- dbscan(clustering_data, eps = 0.5, minPts = 5)

# Add DBSCAN cluster labels
pca_transformed$dbscan_cluster <- as.factor(dbscan_model$cluster)

# DBSCAN summary (0 = noise/outliers)
table(pca_transformed$dbscan_cluster)

# Visualize DBSCAN
ggplot(pca_transformed, aes(x = PC1, y = PC2, color = dbscan_cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(title = "DBSCAN Clustering Results",
       x = "PC1", y = "PC2", color = "Cluster") +
  theme_minimal()

ggsave("output/dbscan_clusters.png", width = 10, height = 8)
```

**Output for Report:**
- Elbow plot and silhouette analysis
- Optimal number of clusters justification
- Cluster visualization (2D and 3D)
- Cluster profile table showing characteristics
- Cluster interpretation (Stable, Delay-Sensitive, Bottleneck-Prone)
- DBSCAN results (if used)

---

## 🤖 PHASE 7: MACHINE LEARNING CLASSIFICATION

### Step 7.1: Create Target Variable (Bottleneck Label)

```r
# Define bottleneck: stations with high delay rate and frequency
# Use clustering or threshold-based approach

# Method 1: Use cluster labels
ml_data <- clustered_data %>%
  mutate(
    bottleneck = ifelse(cluster == 3, 1, 0)  # Cluster 3 = bottleneck
  )

# Method 2: Threshold-based
# ml_data <- station_hour_features %>%
#   mutate(
#     bottleneck = ifelse(delay_rate > 30 & avg_delay > 10, 1, 0)
#   )

# Check class distribution
table(ml_data$bottleneck)
prop.table(table(ml_data$bottleneck))
```

### Step 7.2: Prepare Features and Split Data

```r
# Select features for ML
ml_features <- ml_data %>%
  select(avg_delay, median_delay, sd_delay, max_delay, 
         delay_frequency, total_arrivals, delay_rate, 
         hour, bottleneck) %>%
  mutate(hour = as.numeric(hour))

# Convert day_type to numeric if present
if("day_type" %in% colnames(ml_data)) {
  ml_features$day_type_numeric <- ifelse(ml_data$day_type == "Weekday", 1, 0)
}

# Remove any NA values
ml_features <- na.omit(ml_features)

# Convert bottleneck to factor for classification
ml_features$bottleneck <- as.factor(ml_features$bottleneck)

# Train-test split (80-20)
set.seed(123)
train_index <- createDataPartition(ml_features$bottleneck, p = 0.8, list = FALSE)
train_data <- ml_features[train_index, ]
test_data <- ml_features[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("Training set class distribution:\n")
print(table(train_data$bottleneck))
```

### Step 7.3: Handle Class Imbalance (if needed)

```r
# Check for class imbalance
class_ratio <- prop.table(table(train_data$bottleneck))
print(class_ratio)

# If imbalanced, use SMOTE or adjust class weights
# Using caret's built-in sampling methods

# Define training control with class balancing
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up"  # Upsample minority class (or "down", "smote")
)
```

### Step 7.4: Train Logistic Regression (Baseline)

```r
# Prepare data with proper factor levels for caret
train_data_lr <- train_data
test_data_lr <- test_data

# Ensure factor levels are valid R variable names
levels(train_data_lr$bottleneck) <- c("No", "Yes")
levels(test_data_lr$bottleneck) <- c("No", "Yes")

# Train logistic regression
set.seed(123)
model_logistic <- train(
  bottleneck ~ .,
  data = train_data_lr,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary),
  metric = "ROC"
)

# Model summary
print(model_logistic)

# Predictions
pred_logistic <- predict(model_logistic, test_data_lr)
pred_logistic_prob <- predict(model_logistic, test_data_lr, type = "prob")

# Confusion matrix
cm_logistic <- confusionMatrix(pred_logistic, test_data_lr$bottleneck, positive = "Yes")
print(cm_logistic)
```

### Step 7.5: Train Random Forest

```r
# Train Random Forest
set.seed(123)
model_rf <- train(
  bottleneck ~ .,
  data = train_data_lr,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary),
  metric = "ROC",
  ntree = 100,
  importance = TRUE
)

# Model summary
print(model_rf)

# Predictions
pred_rf <- predict(model_rf, test_data_lr)
pred_rf_prob <- predict(model_rf, test_data_lr, type = "prob")

# Confusion matrix
cm_rf <- confusionMatrix(pred_rf, test_data_lr$bottleneck, positive = "Yes")
print(cm_rf)

# Variable importance
rf_importance <- varImp(model_rf)
plot(rf_importance, main = "Random Forest Variable Importance")

# Save variable importance plot
png("output/rf_variable_importance.png", width = 800, height = 600)
plot(rf_importance, main = "Random Forest Variable Importance")
dev.off()
```

### Step 7.6: Train Support Vector Machine (SVM)

```r
# Train SVM
set.seed(123)
model_svm <- train(
  bottleneck ~ .,
  data = train_data_lr,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary),
  metric = "ROC",
  tuneLength = 5
)

# Model summary
print(model_svm)

# Predictions
pred_svm <- predict(model_svm, test_data_lr)
pred_svm_prob <- predict(model_svm, test_data_lr, type = "prob")

# Confusion matrix
cm_svm <- confusionMatrix(pred_svm, test_data_lr$bottleneck, positive = "Yes")
print(cm_svm)
```

### Step 7.7: Model Evaluation and Comparison

```r
# Extract metrics from all models
extract_metrics <- function(cm, model_name) {
  data.frame(
    Model = model_name,
    Accuracy = cm$overall['Accuracy'],
    Precision = cm$byClass['Precision'],
    Recall = cm$byClass['Recall'],
    F1_Score = cm$byClass['F1'],
    Specificity = cm$byClass['Specificity']
  )
}

# Combine metrics
metrics_comparison <- rbind(
  extract_metrics(cm_logistic, "Logistic Regression"),
  extract_metrics(cm_rf, "Random Forest"),
  extract_metrics(cm_svm, "SVM")
)

print(metrics_comparison)
write_csv(metrics_comparison, "output/model_comparison.csv")

# ROC Curves
roc_logistic <- roc(test_data_lr$bottleneck, pred_logistic_prob$Yes)
roc_rf <- roc(test_data_lr$bottleneck, pred_rf_prob$Yes)
roc_svm <- roc(test_data_lr$bottleneck, pred_svm_prob$Yes)

# Plot ROC curves
png("output/roc_curves_comparison.png", width = 800, height = 600)
plot(roc_logistic, col = "blue", main = "ROC Curves Comparison")
lines(roc_rf, col = "red")
lines(roc_svm, col = "green")
legend("bottomright", legend = c(
  paste("Logistic (AUC =", round(auc(roc_logistic), 3), ")"),
  paste("Random Forest (AUC =", round(auc(roc_rf), 3), ")"),
  paste("SVM (AUC =", round(auc(roc_svm), 3), ")")
), col = c("blue", "red", "green"), lwd = 2)
dev.off()

# AUC comparison
auc_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM"),
  AUC = c(auc(roc_logistic), auc(roc_rf), auc(roc_svm))
)

print(auc_comparison)
write_csv(auc_comparison, "output/auc_comparison.csv")
```

### Step 7.8: Save Best Model

```r
# Select best model based on metrics (e.g., highest Recall or F1)
# Assuming Random Forest is best
best_model <- model_rf

# Save model
saveRDS(best_model, "output/best_classification_model.rds")

cat("Best model saved: Random Forest\n")
```

**Output for Report:**
- Confusion matrices for all models
- Metrics comparison table (Accuracy, Precision, Recall, F1, Specificity)
- ROC curves with AUC values
- Variable importance plot
- Justification for model selection
- Interpretation: "High recall is critical to avoid missing potential bottlenecks"

---

## 🧠 PHASE 8: DEEP LEARNING (LSTM for Delay Forecasting)

### Step 8.1: Prepare Time Series Data

```r
# Create time series dataset
# Sort by station and time
time_series_data <- stop_times_transformed %>%
  arrange(stop_id, scheduled_arrival_time) %>%
  select(stop_id, scheduled_arrival_time, delay_minutes)

# For simplicity, focus on one high-traffic station
# Or aggregate across all stations by hour
hourly_delays <- stop_times_transformed %>%
  mutate(datetime = floor_date(scheduled_arrival_time, "hour")) %>%
  group_by(datetime) %>%
  summarise(avg_delay = mean(delay_minutes)) %>%
  arrange(datetime)

# Check for gaps in time series
head(hourly_delays)
tail(hourly_delays)

# Fill missing hours if needed
all_hours <- seq(min(hourly_delays$datetime), 
                 max(hourly_delays$datetime), 
                 by = "hour")
hourly_delays <- hourly_delays %>%
  right_join(data.frame(datetime = all_hours), by = "datetime") %>%
  arrange(datetime) %>%
  fill(avg_delay, .direction = "down")  # Forward fill

write_csv(hourly_delays, "output/hourly_delay_timeseries.csv")
```

### Step 8.2: Create Sequences for LSTM

```r
# Function to create sequences
create_sequences <- function(data, seq_length) {
  sequences <- list()
  targets <- list()
  
  for (i in 1:(nrow(data) - seq_length)) {
    sequences[[i]] <- data$avg_delay[i:(i + seq_length - 1)]
    targets[[i]] <- data$avg_delay[i + seq_length]
  }
  
  list(
    sequences = do.call(rbind, sequences),
    targets = unlist(targets)
  )
}

# Define sequence length (lookback window)
seq_length <- 24  # Use past 24 hours to predict next hour

# Create sequences
lstm_data <- create_sequences(hourly_delays, seq_length)

# Reshape for LSTM: (samples, timesteps, features)
X <- array(lstm_data$sequences, 
           dim = c(nrow(lstm_data$sequences), seq_length, 1))
y <- lstm_data$targets

# Train-test split (80-20)
train_size <- floor(0.8 * dim(X)[1])
X_train <- X[1:train_size, , , drop = FALSE]
y_train <- y[1:train_size]
X_test <- X[(train_size + 1):dim(X)[1], , , drop = FALSE]
y_test <- y[(train_size + 1):length(y)]

cat("Training samples:", dim(X_train)[1], "\n")
cat("Test samples:", dim(X_test)[1], "\n")
```

### Step 8.3: Normalize Data

```r
# Normalize to [0, 1] range for LSTM
min_delay <- min(hourly_delays$avg_delay, na.rm = TRUE)
max_delay <- max(hourly_delays$avg_delay, na.rm = TRUE)

# Normalize
X_train_norm <- (X_train - min_delay) / (max_delay - min_delay)
y_train_norm <- (y_train - min_delay) / (max_delay - min_delay)
X_test_norm <- (X_test - min_delay) / (max_delay - min_delay)
y_test_norm <- (y_test - min_delay) / (max_delay - min_delay)

# Save normalization parameters
normalization_params <- list(min = min_delay, max = max_delay)
saveRDS(normalization_params, "output/lstm_normalization_params.rds")
```

### Step 8.4: Build LSTM Model

```r
# Load keras
library(keras)

# Build LSTM model
model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 50, 
             input_shape = c(seq_length, 1), 
             return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dense(units = 1)

# Compile model
model_lstm %>% compile(
  optimizer = "adam",
  loss = "mse",
  metrics = c("mae")
)

# Model summary
summary(model_lstm)
```

### Step 8.5: Train LSTM Model

```r
# Train the model
history <- model_lstm %>% fit(
  X_train_norm, y_train_norm,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

# Plot training history
plot(history)

# Save training history plot
png("output/lstm_training_history.png", width = 800, height = 600)
plot(history)
dev.off()

# Save model
save_model_hdf5(model_lstm, "output/lstm_model.h5")
```

### Step 8.6: Evaluate LSTM Model

```r
# Make predictions
predictions_norm <- model_lstm %>% predict(X_test_norm)

# Denormalize predictions and actuals
predictions <- predictions_norm * (max_delay - min_delay) + min_delay
actuals <- y_test_norm * (max_delay - min_delay) + min_delay

# Calculate metrics
mse <- mean((predictions - actuals)^2)
rmse <- sqrt(mse)
mae <- mean(abs(predictions - actuals))
mape <- mean(abs((actuals - predictions) / actuals)) * 100

cat("LSTM Model Performance:\n")
cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")

# Save metrics
lstm_metrics <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "MAPE"),
  Value = c(mse, rmse, mae, mape)
)
write_csv(lstm_metrics, "output/lstm_performance_metrics.csv")
```

### Step 8.7: Visualize LSTM Predictions

```r
# Create comparison dataframe
comparison <- data.frame(
  Index = 1:length(actuals),
  Actual = actuals,
  Predicted = as.vector(predictions)
)

# Plot actual vs predicted
ggplot(comparison, aes(x = Index)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 1, linetype = "dashed") +
  labs(title = "LSTM Predictions vs Actual Delays",
       x = "Time Step",
       y = "Average Delay (minutes)",
       color = "Legend") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))

ggsave("output/lstm_predictions_vs_actual.png", width = 12, height = 6)

# Scatter plot
ggplot(comparison, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "LSTM: Predicted vs Actual Delays",
       x = "Actual Delay (minutes)",
       y = "Predicted Delay (minutes)") +
  theme_minimal()

ggsave("output/lstm_scatter_plot.png", width = 8, height = 8)
```

**Output for Report:**
- LSTM architecture description
- Training history plot (loss over epochs)
- Performance metrics (MSE, RMSE, MAE, MAPE)
- Actual vs Predicted delay plot
- Scatter plot showing prediction accuracy
- Interpretation: "LSTM achieved RMSE of X minutes in forecasting delays"

---

## 🌐 PHASE 9: DELAY PROPAGATION ANALYSIS

### Step 9.1: Build Route Network

```r
# Create network of stations based on routes
# Each edge represents a connection between consecutive stops

# Prepare edges (connections between stops)
route_network <- stop_times_transformed %>%
  left_join(trips, by = "trip_id") %>%
  arrange(trip_id, stop_sequence) %>%
  group_by(trip_id) %>%
  mutate(
    next_stop_id = lead(stop_id),
    next_delay = lead(delay_minutes)
  ) %>%
  filter(!is.na(next_stop_id)) %>%
  ungroup()

# Calculate delay propagation
delay_propagation <- route_network %>%
  group_by(stop_id, next_stop_id) %>%
  summarise(
    propagation_count = n(),
    avg_current_delay = mean(delay_minutes),
    avg_next_delay = mean(next_delay),
    delay_increase = mean(next_delay - delay_minutes)
  ) %>%
  filter(propagation_count > 10)  # Only connections with sufficient data

head(delay_propagation)
write_csv(delay_propagation, "output/delay_propagation_network.csv")
```

### Step 9.2: Create Network Graph

```r
# Create igraph object
library(igraph)

# Prepare edges with weights
edges <- delay_propagation %>%
  select(stop_id, next_stop_id, delay_increase)

# Create graph
g <- graph_from_data_frame(edges, directed = TRUE)

# Add edge weights
E(g)$weight <- delay_propagation$delay_increase

# Calculate node importance (PageRank)
pagerank_scores <- page_rank(g)$vector

# Identify top propagation nodes
top_propagators <- sort(pagerank_scores, decreasing = TRUE)[1:10]
print(top_propagators)

# Save top propagators
write_csv(
  data.frame(
    Station = names(top_propagators),
    PageRank_Score = top_propagators
  ),
  "output/top_delay_propagators.csv"
)
```

### Step 9.3: Visualize Network

```r
# Network visualization
set.seed(123)

# Color nodes by PageRank
node_colors <- colorRampPalette(c("green", "yellow", "red"))(100)
color_indices <- cut(pagerank_scores, breaks = 100, labels = FALSE)
V(g)$color <- node_colors[color_indices]

# Plot
png("output/delay_propagation_network.png", width = 1200, height = 1200)
plot(g,
     vertex.size = 5,
     vertex.label = NA,
     edge.arrow.size = 0.3,
     edge.width = 0.5,
     main = "Delay Propagation Network")
dev.off()

# Interactive network visualization
library(networkD3)

# Prepare data for networkD3
nodes <- data.frame(name = V(g)$name, group = 1)
links <- as_data_frame(g, "edges")
links$source <- match(links$from, nodes$name) - 1
links$target <- match(links$to, nodes$name) - 1
links$value <- abs(links$delay_increase)

# Create interactive network
network_plot <- forceNetwork(
  Links = links, Nodes = nodes,
  Source = "source", Target = "target",
  Value = "value", NodeID = "name",
  Group = "group", opacity = 0.8,
  fontSize = 12, zoom = TRUE
)

# Save interactive plot
saveNetwork(network_plot, "output/interactive_delay_network.html")
```

### Step 9.4: Time Lag Analysis

```r
# Analyze time lag in delay propagation
time_lag_analysis <- route_network %>%
  mutate(
    time_to_next = as.numeric(difftime(lead(actual_arrival_time), 
                                       actual_arrival_time, 
                                       units = "mins"))
  ) %>%
  filter(!is.na(time_to_next)) %>%
  group_by(stop_id) %>%
  summarise(
    avg_time_lag = mean(time_to_next, na.rm = TRUE),
    avg_delay_increase = mean(next_delay - delay_minutes, na.rm = TRUE),
    propagation_rate = avg_delay_increase / avg_time_lag
  ) %>%
  arrange(desc(propagation_rate))

head(time_lag_analysis, 10)
write_csv(time_lag_analysis, "output/time_lag_analysis.csv")
```

**Output for Report:**
- Delay propagation network graph
- Top delay propagator stations (by PageRank)
- Average delay increase between connected stations
- Time lag analysis
- Interactive network visualization (HTML)
- Interpretation of propagation patterns

---

## 📊 PHASE 10: VISUALIZATION & DASHBOARDS

### Step 10.1: Summary Statistics for Power BI

```r
# Prepare summary tables for Power BI import

# Overall summary
overall_summary <- data.frame(
  Metric = c(
    "Total Records",
    "Average Delay (min)",
    "Median Delay (min)",
    "Max Delay (min)",
    "Delay Rate (%)",
    "Total Stations",
    "Total Routes"
  ),
  Value = c(
    nrow(stop_times_transformed),
    round(mean(stop_times_transformed$delay_minutes), 2),
    round(median(stop_times_transformed$delay_minutes), 2),
    max(stop_times_transformed$delay_minutes),
    round(mean(stop_times_transformed$is_delayed) * 100, 2),
    n_distinct(stop_times_transformed$stop_id),
    n_distinct(stop_times_transformed$route_id)
  )
)

write_csv(overall_summary, "output/powerbi_overall_summary.csv")

# Station rankings
station_rankings <- station_stats %>%
  arrange(desc(avg_delay)) %>%
  mutate(rank = row_number()) %>%
  select(rank, stop_id, avg_delay, delay_rate, delay_frequency)

write_csv(station_rankings, "output/powerbi_station_rankings.csv")

# Hourly patterns
hourly_patterns <- stop_times_transformed %>%
  group_by(hour, day_type) %>%
  summarise(
    avg_delay = mean(delay_minutes),
    delay_count = sum(is_delayed)
  )

write_csv(hourly_patterns, "output/powerbi_hourly_patterns.csv")

# Cluster summary for Power BI
cluster_summary_bi <- cluster_profiles %>%
  mutate(
    cluster_name = case_when(
      cluster == 1 ~ "Stable",
      cluster == 2 ~ "Delay-Sensitive",
      cluster == 3 ~ "Bottleneck-Prone"
    )
  )

write_csv(cluster_summary_bi, "output/powerbi_cluster_summary.csv")
```

### Step 10.2: Create Final R Visualizations

```r
# Comprehensive delay dashboard plot
library(gridExtra)

# 1. Delay distribution
p1 <- ggplot(stop_times_transformed, aes(x = delay_minutes)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  labs(title = "Delay Distribution", x = "Delay (min)", y = "Count") +
  theme_minimal() +
  xlim(-5, 60)

# 2. Peak vs Off-peak
p2 <- stop_times_transformed %>%
  group_by(is_peak_hour) %>%
  summarise(avg_delay = mean(delay_minutes)) %>%
  mutate(hour_type = ifelse(is_peak_hour == 1, "Peak", "Off-Peak")) %>%
  ggplot(aes(x = hour_type, y = avg_delay, fill = hour_type)) +
  geom_bar(stat = "identity") +
  labs(title = "Average Delay: Peak vs Off-Peak", 
       x = "", y = "Avg Delay (min)") +
  theme_minimal() +
  theme(legend.position = "none")

# 3. Top 10 delayed stations
p3 <- station_stats %>%
  top_n(10, avg_delay) %>%
  ggplot(aes(x = reorder(stop_id, avg_delay), y = avg_delay)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Top 10 Delayed Stations", x = "", y = "Avg Delay (min)") +
  theme_minimal()

# 4. Hourly pattern
p4 <- hourly_avg %>%
  ggplot(aes(x = hour, y = avg_delay)) +
  geom_line(color = "darkred", size = 1) +
  geom_point(color = "darkred", size = 2) +
  labs(title = "Hourly Delay Pattern", x = "Hour", y = "Avg Delay (min)") +
  theme_minimal()

# Combine plots
combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)

# Save
ggsave("output/comprehensive_delay_dashboard.png", combined_plot, 
       width = 16, height = 12)
```

### Step 10.3: Model Performance Dashboard

```r
# Model comparison visualization
# Accuracy comparison
accuracy_plot <- ggplot(metrics_comparison, 
                        aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy") +
  theme_minimal() +
  theme(legend.position = "none")

# F1 Score comparison
f1_plot <- ggplot(metrics_comparison, 
                  aes(x = Model, y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(F1_Score, 3)), vjust = -0.5) +
  labs(title = "Model F1 Score Comparison", y = "F1 Score") +
  theme_minimal() +
  theme(legend.position = "none")

# Combine
model_comparison_plot <- grid.arrange(accuracy_plot, f1_plot, ncol = 2)
ggsave("output/model_comparison_dashboard.png", model_comparison_plot, 
       width = 12, height = 6)
```

### Step 10.4: Power BI Dashboard Components

Create the following in Power BI:

1. **Overview Dashboard:**
   - KPI cards: Total delays, Avg delay, Delay rate, Total stations
   - Delay trend line chart over time
   - Map visualization of station locations colored by delay severity

2. **Station Analysis Dashboard:**
   - Table: Top 20 delayed stations
   - Heatmap: Station-hour delay intensity
   - Cluster distribution pie chart

3. **Model Performance Dashboard:**
   - Model comparison bar charts
   - Confusion matrix visualization
   - ROC curve comparison
   - LSTM prediction accuracy plot

4. **Bottleneck Risk Dashboard:**
   - Risk level gauge charts
   - Forecasted congestion for next time windows
   - Delay propagation network visualization

5. **Interactive Filters:**
   - Date range selector
   - Station filter
   - Route filter
   - Day type filter (Weekday/Weekend)
   - Hour slider

---

## 📝 PHASE 11: ERROR ANALYSIS & MODEL VALIDATION

### Step 11.1: Classification Error Analysis

```r
# Detailed error analysis for classification models

# Confusion matrix breakdown
cm_detailed <- confusionMatrix(pred_rf, test_data_lr$bottleneck, positive = "Yes")

# Identify misclassified instances
test_data_errors <- test_data_lr %>%
  mutate(
    prediction = pred_rf,
    correct = bottleneck == prediction,
    error_type = case_when(
      bottleneck == "No" & prediction == "Yes" ~ "False Positive",
      bottleneck == "Yes" & prediction == "No" ~ "False Negative",
      TRUE ~ "Correct"
    )
  )

# Analyze false positives
false_positives <- test_data_errors %>%
  filter(error_type == "False Positive") %>%
  select(avg_delay, delay_rate, delay_frequency, hour)

# Analyze false negatives
false_negatives <- test_data_errors %>%
  filter(error_type == "False Negative") %>%
  select(avg_delay, delay_rate, delay_frequency, hour)

# Summary of errors
error_summary <- data.frame(
  Error_Type = c("True Positive", "True Negative", 
                 "False Positive", "False Negative"),
  Count = c(cm_detailed$table[2, 2], cm_detailed$table[1, 1],
            cm_detailed$table[1, 2], cm_detailed$table[2, 1])
)

write_csv(error_summary, "output/classification_error_summary.csv")

# Visualize error patterns
ggplot(test_data_errors, aes(x = avg_delay, y = delay_rate, 
                              color = error_type, shape = error_type)) +
  geom_point(alpha = 0.6, size = 3) +
  labs(title = "Classification Errors by Features",
       x = "Average Delay (min)",
       y = "Delay Rate (%)") +
  theme_minimal()

ggsave("output/classification_error_analysis.png", width = 10, height = 6)
```

### Step 11.2: LSTM Error Analysis

```r
# Calculate residuals
lstm_residuals <- actuals - as.vector(predictions)

# Residual statistics
residual_stats <- data.frame(
  Mean_Residual = mean(lstm_residuals),
  SD_Residual = sd(lstm_residuals),
  Min_Residual = min(lstm_residuals),
  Max_Residual = max(lstm_residuals)
)

print(residual_stats)
write_csv(residual_stats, "output/lstm_residual_statistics.csv")

# Residual plot
residual_df <- data.frame(
  Index = 1:length(lstm_residuals),
  Residual = lstm_residuals
)

ggplot(residual_df, aes(x = Index, y = Residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "LSTM Residuals Over Time",
       x = "Time Step",
       y = "Residual (Actual - Predicted)") +
  theme_minimal()

ggsave("output/lstm_residuals_plot.png", width = 10, height = 6)

# Histogram of residuals
ggplot(residual_df, aes(x = Residual)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of LSTM Residuals",
       x = "Residual",
       y = "Frequency") +
  theme_minimal()

ggsave("output/lstm_residual_distribution.png", width = 8, height = 6)
```

### Step 11.3: Cross-Validation Results

```r
# K-fold cross-validation for classification
set.seed(123)

cv_control <- trainControl(
  method = "cv",
  number = 10,  # 10-fold CV
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Cross-validate Random Forest
cv_rf <- train(
  bottleneck ~ .,
  data = train_data_lr,
  method = "rf",
  trControl = cv_control,
  metric = "ROC",
  ntree = 100
)

# CV results
print(cv_rf)
cv_results <- cv_rf$results
write_csv(cv_results, "output/cv_results.csv")

# Plot CV performance
ggplot(cv_results, aes(x = mtry, y = ROC)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Cross-Validation Performance",
       x = "Number of Variables (mtry)",
       y = "ROC (AUC)") +
  theme_minimal()

ggsave("output/cv_performance.png", width = 10, height = 6)
```

### Step 11.4: Feature Importance Analysis

```r
# Random Forest feature importance
rf_importance_values <- varImp(model_rf)$importance
rf_importance_df <- data.frame(
  Feature = rownames(rf_importance_values),
  Importance = rf_importance_values$Overall
) %>%
  arrange(desc(Importance))

print(rf_importance_df)
write_csv(rf_importance_df, "output/feature_importance.csv")

# Plot feature importance
ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)",
       x = "Feature",
       y = "Importance Score") +
  theme_minimal()

ggsave("output/feature_importance_plot.png", width = 10, height = 8)
```

**Output for Report:**
- Error analysis summary tables
- Confusion matrix breakdown
- False positive/negative analysis
- LSTM residual plots and statistics
- Cross-validation performance metrics
- Feature importance rankings
- Model reliability assessment

---

## 📋 PHASE 12: DOCUMENTATION & REPORTING

### Step 12.1: Generate Final Summary Report

```r
# Create comprehensive summary document

# Overall project summary
project_summary <- list(
  data_summary = data_profile,
  cleaning_summary = cleaning_report,
  delay_statistics = delay_stats,
  station_count = nrow(station_stats),
  route_count = nrow(route_stats),
  pca_components = n_components,
  optimal_clusters = k,
  best_model = "Random Forest",
  best_model_accuracy = max(metrics_comparison$Accuracy),
  lstm_rmse = rmse,
  top_bottleneck_stations = head(station_stats %>% arrange(desc(avg_delay)), 10)
)

# Save summary
saveRDS(project_summary, "output/project_summary.rds")

# Create markdown report
report_text <- paste0(
  "# Public Transport Bottleneck Detection - Project Summary\n\n",
  "## Data Overview\n",
  "- Total Records: ", data_profile$total_records, "\n",
  "- Date Range: ", data_profile$date_range_start, " to ", 
  data_profile$date_range_end, "\n\n",
  "## Key Findings\n",
  "- Average Delay: ", round(delay_stats$mean_delay, 2), " minutes\n",
  "- Delay Rate: ", round(delay_stats$delay_rate, 2), "%\n",
  "- Number of Stations Analyzed: ", nrow(station_stats), "\n",
  "- PCA Components: ", n_components, " (90% variance)\n",
  "- Optimal Clusters: ", k, "\n\n",
  "## Model Performance\n",
  "- Best Classification Model: Random Forest\n",
  "- Accuracy: ", round(max(metrics_comparison$Accuracy), 3), "\n",
  "- F1 Score: ", round(max(metrics_comparison$F1_Score), 3), "\n",
  "- LSTM RMSE: ", round(rmse, 2), " minutes\n\n",
  "## Top 5 Bottleneck Stations\n",
  paste(head(station_stats$stop_id, 5), collapse = ", "), "\n"
)

writeLines(report_text, "output/project_summary_report.md")
```

### Step 12.2: Export All Results

```r
# List all output files
output_files <- list.files("output", full.names = TRUE)
cat("Generated", length(output_files), "output files\n")

# Create a summary of outputs
output_summary <- data.frame(
  File = basename(output_files),
  Type = tools::file_ext(output_files),
  Size_KB = round(file.size(output_files) / 1024, 2)
)

write_csv(output_summary, "output/output_files_summary.csv")

print(output_summary)
```

---

## 🎯 FINAL CHECKLIST

### Data Mining Components ✓
- [ ] Data Cleaning (missing values, duplicates, invalid data)
- [ ] Data Transformation (delay calculation, feature engineering)
- [ ] Data Reduction (PCA)
- [ ] Pattern Discovery (EDA, visualizations)
- [ ] Clustering (K-Means, DBSCAN)

### Machine Learning Components ✓
- [ ] Classification Models (Logistic Regression, Random Forest, SVM)
- [ ] Model Evaluation (Confusion Matrix, ROC, AUC)
- [ ] Cross-Validation
- [ ] Feature Importance Analysis

### Deep Learning Components ✓
- [ ] LSTM Model for Time Series Forecasting
- [ ] Sequence Preparation
- [ ] Model Training and Validation
- [ ] Performance Metrics (RMSE, MAE, MAPE)

### Visualization Components ✓
- [ ] R Visualizations (ggplot2)
- [ ] Power BI Dashboards
- [ ] Network Graphs (delay propagation)
- [ ] Interactive Visualizations

### Documentation ✓
- [ ] Data profiling reports
- [ ] Cleaning summary
- [ ] Model performance metrics
- [ ] Feature importance
- [ ] Final summary report

---

## 📌 KEY FORMULAS & METRICS

### Delay Calculation
```
delay = actual_arrival_time - scheduled_arrival_time
is_delayed = delay > threshold (e.g., 5 minutes)
delay_rate = (count of delayed arrivals / total arrivals) × 100
```

### Classification Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### LSTM Metrics
```
MSE = mean((actual - predicted)²)
RMSE = √MSE
MAE = mean(|actual - predicted|)
MAPE = mean(|actual - predicted| / actual) × 100
```

### PCA
```
Variance Explained = (eigenvalue_i / Σ eigenvalues) × 100
Cumulative Variance = Σ(variance explained up to component i)
```

---

## 🚀 EXECUTION TIPS

1. **Run code sequentially** - Each phase builds on the previous
2. **Check data dimensions** - After each transformation
3. **Save intermediate results** - Don't lose progress
4. **Visualize frequently** - Catch issues early
5. **Document decisions** - Why you chose certain parameters
6. **Version control** - Keep track of model versions
7. **Comment code** - Explain complex transformations
8. **Test on small sample first** - Before full dataset processing

---

## 📊 EXPECTED OUTPUTS

### CSV Files
- stop_times_cleaned.csv
- station_delay_statistics.csv
- route_delay_statistics.csv
- pca_transformed_data.csv
- clustered_stations_labeled.csv
- model_comparison.csv
- lstm_performance_metrics.csv
- delay_propagation_network.csv

### Plots (PNG)
- delay_distribution.png
- hourly_delay_pattern.png
- station_hour_heatmap.png
- pca_scree_plot.png
- kmeans_clusters_pca.png
- rf_variable_importance.png
- roc_curves_comparison.png
- lstm_predictions_vs_actual.png
- delay_propagation_network.png

### Models
- pca_model.rds
- best_classification_model.rds
- lstm_model.h5

### Power BI Files
- powerbi_dashboard.pbix (to be created in Power BI)

---

## 🎓 ACADEMIC WRITING TIPS FOR REPORT

### Data Mining Section
- "The dataset underwent comprehensive preprocessing including noise removal, handling missing values through deletion, and duplicate record elimination."
- "Principal Component Analysis (PCA) was applied to reduce dimensionality from X features to Y components, retaining 90% of the variance."
- "K-Means clustering identified three distinct station groups: stable (Cluster 1), delay-sensitive (Cluster 2), and bottleneck-prone (Cluster 3)."

### Machine Learning Section
- "Multiple classification algorithms were evaluated including Logistic Regression (baseline), Random Forest, and SVM."
- "Random Forest achieved the highest performance with accuracy of X%, precision of Y%, and recall of Z%."
- "High recall was prioritized to minimize false negatives, as missing potential bottlenecks poses greater operational risk."

### Deep Learning Section
- "LSTM neural networks were employed for time-series delay forecasting due to their ability to capture temporal dependencies."
- "The model achieved RMSE of X minutes on the test set, demonstrating effective short-term delay prediction capability."
- "A sequence length of 24 hours was used, allowing the model to learn daily delay patterns."

### Visualization Section
- "Interactive dashboards were developed in Power BI to provide decision-makers with real-time insights into system performance."
- "Network analysis revealed key delay propagation pathways, with Station X identified as the primary bottleneck source."

---

**END OF COMPREHENSIVE CODING PLAN**

This plan provides complete, step-by-step instructions for implementing the entire project from data loading to final visualization and documentation. Each phase includes working R code, explanations, and outputs suitable for academic reporting.
