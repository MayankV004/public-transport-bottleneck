# =============================================================================
# PHASE 9: DELAY PROPAGATION NETWORK ANALYSIS
# Script: 09_propagation.R
# Purpose: Build a station connectivity graph, compute centrality metrics,
#          identify top delay propagation hubs, and visualize the network.
# =============================================================================

cat("=== PHASE 9: DELAY PROPAGATION NETWORK ANALYSIS ===\n\n")

# --- Load libraries -----------------------------------------------------------
library(readr)
library(dplyr)
library(igraph)
library(ggplot2)

# Check for networkD3
networkD3_available <- requireNamespace("networkD3", quietly = TRUE)
if (!networkD3_available) {
  tryCatch({
    install.packages("networkD3", repos = "https://cloud.r-project.org")
    library(networkD3)
    networkD3_available <- TRUE
  }, error = function(e) {
    cat("  networkD3 not available — skipping interactive network viz.\n")
  })
} else {
  library(networkD3)
}

# --- Load transformed data ----------------------------------------------------
df <- readRDS("output/cleaned/transport_delays_transformed.rds")
cat(sprintf("Loaded: %d rows x %d columns\n\n", nrow(df), ncol(df)))

# =============================================================================
# Step 9.1: Build Station Connectivity Graph
# =============================================================================
cat("Step 9.1: Building station connectivity edges...\n")

# Create edges between origin → destination station pairs
edges <- df %>%
  group_by(origin_station, destination_station) %>%
  summarise(
    trip_count          = n(),
    avg_arrival_delay   = round(mean(actual_arrival_delay_min, na.rm = TRUE), 2),
    avg_dep_delay       = round(mean(actual_departure_delay_min, na.rm = TRUE), 2),
    avg_delay_change    = round(mean(delay_change, na.rm = TRUE), 2),
    delay_amplification = round(mean(delay_change[delay_change > 0], na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  filter(origin_station != destination_station)  # Remove self-loops

# Replace NaN with 0
edges$delay_amplification[is.nan(edges$delay_amplification)] <- 0

cat(sprintf("  Created %d edges between %d unique station pairs\n",
            nrow(edges),
            n_distinct(c(edges$origin_station, edges$destination_station))))

write_csv(edges, "output/reports/delay_propagation_edges.csv")
cat("  Saved: output/reports/delay_propagation_edges.csv\n")

# =============================================================================
# Step 9.2: Construct igraph Network
# =============================================================================
cat("\nStep 9.2: Constructing igraph network...\n")

# Create directed graph
g <- graph_from_data_frame(
  d = edges %>% select(origin_station, destination_station, trip_count,
                        avg_arrival_delay, avg_delay_change, delay_amplification),
  directed = TRUE
)

# Set edge weights as average delay
E(g)$weight <- abs(edges$avg_arrival_delay) + 0.1  # +0.1 to avoid zero weights

cat(sprintf("  Network: %d nodes, %d edges\n", vcount(g), ecount(g)))

# Compute node-level metrics
V(g)$degree     <- degree(g, mode = "all")
V(g)$in_degree  <- degree(g, mode = "in")
V(g)$out_degree <- degree(g, mode = "out")
V(g)$pagerank   <- page_rank(g, weights = E(g)$weight)$vector
V(g)$betweenness <- betweenness(g, normalized = TRUE)
V(g)$closeness   <- closeness(g, mode = "all", normalized = TRUE)

# Create node metrics table
node_metrics <- data.frame(
  Station     = V(g)$name,
  Degree      = V(g)$degree,
  In_Degree   = V(g)$in_degree,
  Out_Degree  = V(g)$out_degree,
  PageRank    = round(V(g)$pagerank, 6),
  Betweenness = round(V(g)$betweenness, 6),
  Closeness   = round(V(g)$closeness, 6),
  stringsAsFactors = FALSE
) %>% arrange(desc(PageRank))

cat("\n  Top 10 stations by PageRank:\n")
print(head(node_metrics, 10), row.names = FALSE)

write_csv(node_metrics, "output/reports/station_network_metrics.csv")
cat("  Saved: output/reports/station_network_metrics.csv\n")

# =============================================================================
# Step 9.3: Identify Top Propagation Bottlenecks
# =============================================================================
cat("\nStep 9.3: Identifying top delay propagation bottlenecks...\n")

top10_propagators <- head(node_metrics, 10)
cat("  Top 10 delay propagation hubs (by PageRank):\n")
for (i in 1:nrow(top10_propagators)) {
  cat(sprintf("    %2d. %s (PageRank=%.5f, Degree=%d, Betweenness=%.5f)\n",
              i, top10_propagators$Station[i], top10_propagators$PageRank[i],
              top10_propagators$Degree[i], top10_propagators$Betweenness[i]))
}

# Cross-reference with cluster labels if available
tryCatch({
  df_clustered <- readRDS("output/cleaned/df_with_clusters.rds")
  
  station_cluster <- df_clustered %>%
    group_by(origin_station, cluster) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(origin_station) %>%
    slice_max(n, n = 1) %>%
    select(origin_station, dominant_cluster = cluster)
  
  top10_with_cluster <- top10_propagators %>%
    left_join(station_cluster, by = c("Station" = "origin_station"))
  
  cat("\n  Top propagators with dominant cluster:\n")
  print(top10_with_cluster[, c("Station", "PageRank", "dominant_cluster")], row.names = FALSE)
}, error = function(e) {
  cat("  Could not cross-reference with cluster data.\n")
})

# =============================================================================
# Step 9.4: Visualize the Network
# =============================================================================
cat("\nStep 9.4: Visualizing delay propagation network...\n")

# Static network graph
# Scale node size by PageRank and color by betweenness
node_size <- 5 + (V(g)$pagerank / max(V(g)$pagerank)) * 20
node_color_val <- V(g)$betweenness / max(V(g)$betweenness + 1e-10)

# Color gradient from blue (low) to red (high betweenness)
node_colors <- colorRampPalette(c("lightblue", "orange", "red"))(100)
node_col_idx <- pmax(1, ceiling(node_color_val * 99) + 1)
V(g)$color <- node_colors[node_col_idx]

# Edge width by trip count
edge_width <- 0.5 + (E(g)$weight / max(E(g)$weight)) * 3

png("output/plots/delay_propagation_network.png", width = 1400, height = 1200, res = 150)
set.seed(42)
plot(g,
     vertex.size    = node_size,
     vertex.color   = V(g)$color,
     vertex.label   = gsub("Station_", "S", V(g)$name),
     vertex.label.cex = 0.6,
     vertex.label.color = "black",
     edge.arrow.size = 0.3,
     edge.width     = edge_width,
     edge.color     = adjustcolor("gray50", alpha = 0.5),
     layout         = layout_with_fr(g),
     main           = "Delay Propagation Network\nNode size=PageRank, Color=Betweenness")
dev.off()
cat("  Saved: output/plots/delay_propagation_network.png\n")

# Top propagators bar chart
p_prop <- ggplot(top10_propagators, aes(x = reorder(Station, PageRank),
                                         y = PageRank, fill = PageRank)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red") +
  labs(title = "Top 10 Delay Propagation Hubs (PageRank)",
       x = "Station", y = "PageRank Score") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
ggsave("output/plots/top_propagators_pagerank.png", p_prop,
       width = 10, height = 6, dpi = 300)
cat("  Saved: output/plots/top_propagators_pagerank.png\n")

# Interactive network (networkD3)
if (networkD3_available) {
  cat("  Creating interactive network...\n")
  
  # Prepare data for networkD3 (requires 0-indexed node IDs)
  nodes_d3 <- data.frame(
    name  = V(g)$name,
    group = ceiling(V(g)$pagerank / max(V(g)$pagerank) * 5),
    size  = V(g)$degree
  )
  
  edges_for_d3 <- edges %>%
    mutate(
      source = match(origin_station, nodes_d3$name) - 1,
      target = match(destination_station, nodes_d3$name) - 1,
      value  = trip_count
    ) %>%
    filter(!is.na(source) & !is.na(target))
  
  network_d3 <- forceNetwork(
    Links  = as.data.frame(edges_for_d3),
    Nodes  = nodes_d3,
    Source = "source",
    Target = "target",
    Value  = "value",
    NodeID = "name",
    Group  = "group",
    Nodesize = "size",
    opacity = 0.9,
    zoom    = TRUE,
    fontSize = 12
  )
  
  htmlwidgets::saveWidget(network_d3, "output/plots/interactive_delay_network.html",
                          selfcontained = FALSE)
  cat("  Saved: output/plots/interactive_delay_network.html\n")
}

# =============================================================================
# Step 9.5: Save Outputs
# =============================================================================
cat("\nStep 9.5: Saving network outputs...\n")

saveRDS(g, "output/models/delay_network_graph.rds")
cat("  Saved: output/models/delay_network_graph.rds\n")

cat("\n=== PHASE 9 COMPLETE ===\n")
