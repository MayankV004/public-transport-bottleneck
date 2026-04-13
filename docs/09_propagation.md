# Phase 9: Delay Propagation Network Analysis (`09_propagation.R`)

## Purpose
Models the transport network as a **directed graph** where stations are nodes and routes are edges. Uses graph-theoretic centrality metrics (PageRank, betweenness, closeness) to identify **propagation hubs** — stations where delays originate and spread through the connected system.

---

## Why Network Analysis?

Traditional analysis treats each trip in isolation: "Trip TRP_5541 was delayed 12 minutes." But transport is a *connected system*:

```
Scenario: Train from Station_03 arrives 15 minutes late.
→ 40 passengers miss their connection to Bus at Station_07
→ Bus waits 3 extra minutes to let them board
→ Now the Bus is 3 minutes late to Station_12
→ Passengers at Station_12 miss their Metro
→ Cascade continues through 4 more stations...
```

Network analysis is the only tool that models these *multi-hop cascade effects*.

---

## Step-by-Step Breakdown

### Step 9.1: Build Edge Table
Every unique origin → destination pair becomes a directed edge with aggregate delay statistics.

```r
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
  filter(origin_station != destination_station)  # Exclude self-loops
```

**Example edge table (first 5 rows):**
```
origin_station  dest_station  trip_count  avg_arrival_delay  delay_amplification
   Station_03   Station_07          312               8.41                 5.23
   Station_07   Station_12          287               3.12                 2.10
   Station_03   Station_15          198              14.72                11.40  ← High amplification!
   Station_15   Station_22          410               6.80                 3.50
```

`delay_amplification = 11.40` on Station_03 → Station_15 means: when a trip on that route IS delayed, the delay grows by an average of 11.4 minutes during transit. This edge actively *worsens* delays.

---

### Step 9.2: Construct igraph Network

```r
g <- graph_from_data_frame(
  d        = edges,
  directed = TRUE     # Direction matters: Station_03 → Station_07 ≠ Station_07 → Station_03
)

# Edge weights = average delay (higher delay = stronger influence channel)
E(g)$weight <- abs(edges$avg_arrival_delay) + 0.1  # +0.1 avoids zero weights

cat(sprintf("Network: %d nodes, %d edges\n", vcount(g), ecount(g)))
# Example: "Network: 85 nodes, 1,240 edges"
```

**Four centrality metrics — why each matters:**

```r
V(g)$degree      <- degree(g, mode = "all")       # Total connections
V(g)$in_degree   <- degree(g, mode = "in")        # How many stations feed INTO this one
V(g)$out_degree  <- degree(g, mode = "out")       # How many stations this one feeds OUT TO
V(g)$pagerank    <- page_rank(g, weights = E(g)$weight)$vector  # Influence score
V(g)$betweenness <- betweenness(g, normalized = TRUE)  # Chokepoint score
V(g)$closeness   <- closeness(g, mode = "all", normalized = TRUE) # Spread speed score
```

**Metric Interpretation Table:**

| Metric | High Value Means | Example Station |
|---|---|---|
| **Degree** | Many direct connections | Major interchange hub |
| **In-Degree** | Many routes arriving | Terminal / destination hub |
| **Out-Degree** | Many routes departing | Origin / transfer hub |
| **PageRank** | Receives delays from *important* stations | Network amplifier |
| **Betweenness** | Sits on shortest path between many pairs | Chokepoint — if it fails, rerouting is hard |
| **Closeness** | Can reach all other stations quickly | Systemic risk station |

---

### Step 9.3: Top Propagation Bottlenecks

```r
node_metrics <- data.frame(
  Station     = V(g)$name,
  PageRank    = round(V(g)$pagerank, 6),
  Betweenness = round(V(g)$betweenness, 6),
  Degree      = V(g)$degree
) %>% arrange(desc(PageRank))

cat("Top 10 stations by PageRank:\n")
print(head(node_metrics, 10), row.names = FALSE)
```

**Example output:**
```
     Station  PageRank  Betweenness  Degree
 Station_07  0.04821       0.2341      28   ← Highest influence + major chokepoint
 Station_03  0.03982       0.1987      24
 Station_15  0.03541       0.3102      19   ← Highest betweenness! Critical chokepoint
 Station_22  0.02890       0.1544      21
 Station_44  0.02711       0.0892      33   ← Many connections but low betweenness
```

**Key insight from this table:**
- `Station_15` has *lower* PageRank than `Station_07`, but *higher* betweenness — meaning it's a critical chokepoint even though it's not the most "popular" station. If `Station_15` is disrupted, alternative routes are much longer.

---

### Step 9.4: Visualize the Network

**Static network visualisation:**
```r
# Node size encodes influence (PageRank): bigger node = more influential
node_size <- 5 + (V(g)$pagerank / max(V(g)$pagerank)) * 20

# Node colour encodes chokepoint risk (betweenness): blue → orange → red
node_color_val <- V(g)$betweenness / max(V(g)$betweenness + 1e-10)
node_colors    <- colorRampPalette(c("lightblue", "orange", "red"))(100)
V(g)$color     <- node_colors[pmax(1, ceiling(node_color_val * 99) + 1)]

# Edge width scales with delay severity (thicker = worse delays on that route)
edge_width <- 0.5 + (E(g)$weight / max(E(g)$weight)) * 3

# Fruchterman-Reingold layout: groups tightly-connected nodes naturally
plot(g, layout = layout_with_fr(g),
     vertex.size = node_size, vertex.color = V(g)$color,
     main = "Delay Propagation Network\nNode size=PageRank, Color=Betweenness")
```

**Reading the network plot:**
- 🔴 Large red node = High-influence, high-chokepoint station → immediate operational priority
- 🟡 Medium yellow node = Moderate hub — monitor during peak hours
- 🔵 Small blue node = Low-risk peripheral station

**Interactive D3 network:**
```r
# Creates a zoomable, clickable HTML network (output/plots/interactive_delay_network.html)
network_d3 <- forceNetwork(
  Links = edges_for_d3, Nodes = nodes_d3,
  Source = "source", Target = "target", Value = "value",
  NodeID = "name",   Group = "group",   Nodesize = "size",
  opacity = 0.9, zoom = TRUE, fontSize = 12
)
```

---

## Outputs
| File | Description |
|---|---|
| `output/reports/station_network_metrics.csv` | Degree, PageRank, Betweenness, Closeness per station |
| `output/reports/delay_propagation_edges.csv` | Edges with delay amplification statistics |
| `output/plots/delay_propagation_network.png` | Static colour-coded network |
| `output/plots/top_propagators_pagerank.png` | Top 10 hub bar chart (yellow→red gradient) |
| `output/plots/interactive_delay_network.html` | Zoomable interactive D3 network |
| `output/models/delay_network_graph.rds` | Saved igraph object for future analysis |

---

## 💡 Presentation Talking Points
> "We borrowed PageRank — Google's original algorithm for ranking web pages — to rank transport stations. Just as a webpage is important if other important pages link to it, a station is a propagation hub if it receives delays from other already-delayed stations."

> "Betweenness centrality identified Station_15 as a critical chokepoint despite being only the 3rd most connected station. A delay at Station_15 forces passengers through significantly longer alternative paths — a fact invisible to any per-trip analysis."
