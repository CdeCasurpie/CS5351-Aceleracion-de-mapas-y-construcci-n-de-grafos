# ACJ: Urban Graph Acceleration & Simplification Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C.svg?logo=c%2B%2B&logoColor=white)
![PyPI](https://img.shields.io/pypi/v/geojac.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**ACJ** is a high-performance hybrid C++/Python framework for the semantic and topological simplification of large-scale urban street networks. It safely decouples topology from semantic metadata (speed limits, road names, etc.) so that both survive massive graph reductions powered by a CGAL-accelerated C++ core.

---

## Installation

ACJ is published on PyPI and ships with pre-compiled wheels for Linux, macOS (Apple Silicon & Intel), and Windows. No C++ compilation required.

```bash
pip install geojac
```

For GPU-accelerated real-time visualization, install the optional extras:

```bash
pip install "geojac[viz]"
```

> We recommend working inside a virtual environment: `python -m venv venv && source venv/bin/activate`

---

## Quick Start — End-to-End Pipeline

```python
import osmnx as ox
from geojac import UrbanNetwork, ACJTopologicalEvaluator
from geojac import CompressionRatioMetric, SemanticSpeedDistortionMetric

# 1. Fetch a raw street network
G = ox.graph_from_place("Barranco, Lima, Peru", network_type="drive")

# 2. Parse into the ACJ registry
network = UrbanNetwork.from_networkx(G)
print(network)
# <UrbanNetwork | Nodes: 312 (Meta: True) | Edges: 847 (Meta: True)>

# 3. Configure metrics and run the evaluator
metrics = [CompressionRatioMetric(), SemanticSpeedDistortionMetric()]
evaluator = ACJTopologicalEvaluator(network, metrics)
results = evaluator.evaluate()

print(results)
print(evaluator.simplified_network)
```

---

## API Reference

### `UrbanNetwork` — Core Data Container

The central data structure. Stores topology in Pandas DataFrames and semantic attributes in dictionaries, keeping them decoupled for safe C++ operations.

```python
from geojac import UrbanNetwork

# From an OSMnx / NetworkX graph
network = UrbanNetwork.from_networkx(G)

# From raw DataFrames
# nodes_df must have columns: node_id, x, y  (+ any metadata columns)
# edges_df must have columns: segment_id, node_start, node_end  (+ any metadata columns)
network = UrbanNetwork.from_dataframe(nodes_df, edges_df)
```

**Key attributes:**

| Attribute | Type | Description |
|---|---|---|
| `nodes_df` | `pd.DataFrame` | Node topology: `node_id`, `x`, `y` |
| `edges_df` | `pd.DataFrame` | Edge topology: `segment_id`, `node_start`, `node_end` |
| `node_metadata` | `dict` | Per-node semantic attributes keyed by `node_id` |
| `edge_metadata` | `dict` | Per-edge semantic attributes keyed by `segment_id` |

---

### `MapIndex` — Spatial Queries

CGAL-backed spatial index for fast nearest-neighbor point-to-graph assignment. Useful for correlating events (e.g., crime incidents) with street segments or intersections.

```python
from geojac import MapIndex, load_graph
import pandas as pd

graph_data = load_graph(nodes_df, edges_df)
index = MapIndex(graph_data)

# Assign point events to the nearest node
points = pd.DataFrame({'point_id': [1, 2], 'x': [-77.02, -77.03], 'y': [-12.15, -12.14]})
node_assignments = index.assign_to_endpoints(points)
# Returns original DataFrame + 'assigned_node_id' and 'distance' columns

# Assign point events to the nearest edge segment
segment_assignments = index.assign_to_segments(points)
# Returns original DataFrame + 'assigned_segment_id' and 'distance' columns
```

---

### `ACJTopologicalEvaluator` — Simplification Lifecycle

Automates the full pipeline: injects the network into the C++ core, receives the simplified topology with lineage maps, runs semantic resolution, and applies metrics.

```python
from geojac import ACJTopologicalEvaluator, CompressionRatioMetric, SemanticSpeedDistortionMetric

evaluator = ACJTopologicalEvaluator(network, metrics=[
    CompressionRatioMetric(),
    SemanticSpeedDistortionMetric(),
])
results = evaluator.evaluate()

simplified = evaluator.simplified_network  # UrbanNetwork
```

**Built-in metrics:**

| Class | Measures |
|---|---|
| `CompressionRatioMetric` | Node/edge reduction ratio after simplification |
| `SemanticSpeedDistortionMetric` | Speed-limit distribution distortion introduced by edge merging |

---

### Visualization (optional `viz` extra)

GPU-accelerated interactive tools built on VisPy/OpenGL. Requires `pip install "geojac[viz]"`.

```python
from geojac import MapIndex, load_graph
from geojac import render_graph, render_heatmap, render_comparison

graph_data = load_graph(nodes_df, edges_df)
index = MapIndex(graph_data)

# Render the raw street network
render_graph(index)

# Render a heatmap of point assignments over the network
render_heatmap(index, assignments=node_assignments)

# Side-by-side comparison of two networks (e.g., original vs. simplified)
render_comparison(
    index_original, index_simplified,
    title_left="Original", title_right="Simplified"
)
```

**Interactive controls (all render functions):**

| Key | Action |
|---|---|
| `N` | Toggle node layer |
| `L` | Toggle segment layer |
| `G` | Toggle grid |
| `R` | Reset camera view |
| `Q` / `Esc` | Close window |

---

### `load_graph` / `load_map` — Data IO

```python
from geojac import load_graph, load_map

# Build a GraphData object from DataFrames (required input for MapIndex)
graph_data = load_graph(nodes_df, edges_df)

# Load from a file path (GeoJSON, Shapefile, etc.)
graph_data = load_map("path/to/network.geojson")
```

---

## Developer Installation (From Source)

Required: CMake ≥ 3.15, C++17, CGAL, Boost, pybind11.

```bash
# Ubuntu/Debian
sudo apt-get install cmake libcgal-dev libboost-all-dev

# Arch Linux
sudo pacman -S cmake cgal boost pybind11
```

```bash
git clone https://github.com/CdeCasurpie/CS5351-acj.git
cd CS5351-acj
python -m venv venv && source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

---

## Academic Team & Acknowledgements

This framework was developed as part of a thesis research project at the Universidad de Ingeniería y Tecnología (UTEC).

**Lead Researchers & Engineers:**
* Alejandro Calizaya
* Cesar Perales
* Jerimy Sandoval

**Thesis Advisors:**
* **Eric Javier Biagioli** - Academic Director, Department of Computer and Data Science, UTEC ([ORCID: 0009-0000-2027-0647](https://orcid.org/0009-0000-2027-0647))
* **Germain García Zanabria** - Researcher, Artificial Intelligence Research Group (GINIA), UTEC ([ORCID: 0000-0003-3266-9043](https://orcid.org/0000-0003-3266-9043))
