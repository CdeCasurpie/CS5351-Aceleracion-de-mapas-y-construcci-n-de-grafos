# ACJ - Advanced Crime-to-Junction Assignment Library

A high-performance Python library for geospatial analysis and real-time visualization of point-based events on street networks. Combines CGAL computational geometry algorithms with GPU-accelerated rendering for large-scale urban analytics.

## Overview

ACJ provides efficient spatial assignment of point events (crimes, incidents, service requests) to street network nodes using Delaunay triangulation, with real-time interactive visualization capabilities.

**Key Features:**
- Fast spatial queries using CGAL Delaunay triangulation (6.1x faster than brute-force)
- Automatic street network loading from OpenStreetMap via OSMnx
- GPU-accelerated real-time visualization with VisPy/OpenGL
- Standardized pandas DataFrame interface
- Docker-based reproducible environment

**Performance:** Process 100,000 point assignments on 50,000 nodes in 1.3 seconds

**Based on work by:** Alejandro (CGAL integration and spatial indexing)

---

## Installation

### Docker (Recommended)

```bash
git clone <repository-url>
cd pylib

# Build Docker image with all dependencies
make build

# Verify installation
make test

# Run interactive visualization (requires X11)
xhost +local:docker
make example-realtime
```

### Manual Installation (Linux)

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake python3-dev python3-pip
sudo apt-get install libcgal-dev pybind11-dev
sudo apt-get install libspatialindex-dev libgeos-dev libproj-dev
sudo apt-get install libgl1-mesa-glx python3-pyqt5

# Python dependencies
pip install -r requirements.txt

# Compile C++ extension
mkdir build && cd build
cmake .. && make -j$(nproc)
export PYTHONPATH=/path/to/pylib/build:$PYTHONPATH
```

---

## Quick Start

### Basic Usage

```python
import acj
import pandas as pd

# Load graph from DataFrames
nodes = pd.DataFrame({
    'node_id': [0, 1, 2],
    'x': [0.0, 100.0, 200.0],
    'y': [0.0, 0.0, 100.0]
})

segments = pd.DataFrame({
    'segment_id': [0, 1],
    'node_start': [0, 1],
    'node_end': [1, 2],
    'x1': [0.0, 100.0],
    'y1': [0.0, 0.0],
    'x2': [100.0, 200.0],
    'y2': [0.0, 100.0]
})

graph = acj.load_graph(nodes, segments)

# Create spatial index
map_index = acj.MapIndex(graph)

# Assign points to nearest nodes
crimes = pd.DataFrame({
    'point_id': [0, 1, 2],
    'x': [50.0, 150.0, 25.0],
    'y': [5.0, 10.0, 90.0],
    'crime_type': ['robbery', 'assault', 'theft']
})

assignments = map_index.assign_to_endpoints(crimes)
print(assignments[['point_id', 'assigned_node_id', 'distance']])
```

### Real City Example

```python
import acj
import numpy as np

# Load street network from OpenStreetMap
graph = acj.load_map("Cholula, Puebla, Mexico")

# Generate sample points
np.random.seed(42)
n = 1000
x_min, x_max = graph.nodes['x'].min(), graph.nodes['x'].max()
y_min, y_max = graph.nodes['y'].min(), graph.nodes['y'].max()

crimes = pd.DataFrame({
    'point_id': range(n),
    'x': np.random.uniform(x_min, x_max, n),
    'y': np.random.uniform(y_min, y_max, n)
})

# Assign to street network
map_index = acj.MapIndex(graph)
assignments = map_index.assign_to_endpoints(crimes)

# Launch interactive visualization
acj.render_heatmap(map_index, assignments, 
                   title="Crime Density Heatmap")
```

**Visualization Controls:**
- Mouse drag: Pan view
- Mouse wheel: Zoom
- N key: Toggle node visibility
- R key: Reset camera
- Q or ESC: Close window

---

## Data Format

All inputs and outputs use pandas DataFrames with standardized schemas.

### Nodes DataFrame
```python
pd.DataFrame({
    'node_id': int,    # Unique identifier
    'x': float,        # X coordinate (meters, projected CRS)
    'y': float         # Y coordinate (meters, projected CRS)
})
```

### Segments DataFrame
```python
pd.DataFrame({
    'segment_id': int,     # Unique identifier
    'node_start': int,     # Start node_id
    'node_end': int,       # End node_id
    'x1': float, 'y1': float,  # Start coordinates
    'x2': float, 'y2': float   # End coordinates
})
```

### Points DataFrame (Input)
```python
pd.DataFrame({
    'point_id': int,   # Unique identifier
    'x': float,        # X coordinate
    'y': float         # Y coordinate
    # Additional columns preserved in output
})
```

### Assignments DataFrame (Output)
```python
pd.DataFrame({
    'point_id': int,           # From input
    'x': float, 'y': float,    # From input
    'assigned_node_id': int,   # Nearest node
    'distance': float,         # Euclidean distance (meters)
    # All input columns preserved
})
```

**Important:** Coordinates must be in a projected coordinate system (e.g., UTM) in meters, not latitude/longitude.

---

## API Reference

### Data Loading

**`acj.load_graph(nodes_df, segments_df)`**

Load graph from pandas DataFrames.

Returns: `GraphData` object

---

**`acj.load_map(city_name, cache_dir="./cache", network_type="drive")`**

Download street network from OpenStreetMap.

Parameters:
- `city_name` (str): City query string (e.g., "Manhattan, New York")
- `cache_dir` (str): Cache directory path
- `network_type` (str): Network type ("drive", "walk", "bike", "all")

Returns: `GraphData` object

Example:
```python
graph = acj.load_map("Cholula, Puebla, Mexico")
```

---

### Spatial Indexing

**`acj.MapIndex(graph_data)`**

Create CGAL-based spatial index.

**Methods:**

**`assign_to_endpoints(points_df)`**

Assign points to nearest graph nodes.

Complexity: O(M log M) build + O(N log M) query

Returns: DataFrame with assignments and distances

Example:
```python
map_index = acj.MapIndex(graph)
assignments = map_index.assign_to_endpoints(crimes)
```

**`get_render_data(assignments=None)`**

Pre-compute GPU render buffers.

Returns: Dictionary with vertex/color arrays

---

### Visualization

**`acj.render_heatmap(map_index, assignments, title="Crime Density Heatmap")`**

Launch interactive heatmap visualization.

Features:
- Color gradient: white (low) to yellow to orange to red (high)
- Gradient interpolation on street segments
- Real-time zoom/pan with OpenGL acceleration
- Node visibility toggle

Blocks until window is closed.

---

**`acj.render_graph(map_index, title="Street Network")`**

Launch basic network visualization without heatmap coloring.

---

### Utilities

**`acj.simplify_graph(graph_data, threshold_meters=10.0)`**

Simplify graph by merging nearby nodes.

Status: Currently returns input unchanged (pending implementation)

---

## Examples

### example_acj.py

Basic demonstration with synthetic data.

```bash
make example
```

Features:
- Graph creation from DataFrames
- Spatial index construction
- Point assignment
- Result statistics

---

### example_realtime.py

Complete pipeline with real city data.

```bash
xhost +local:docker
make example-realtime
```

Features:
- OpenStreetMap data loading
- Random point generation
- CGAL spatial assignment
- GPU-accelerated visualization

---

## Performance

### Benchmark Results

| Dataset | CGAL Time | Brute-Force Time | Speedup |
|---------|-----------|------------------|---------|
| 100k queries on 50k targets | 1.29s | 7.89s | 6.1x |

Complexity:
- CGAL: O(M log M) + O(N log M)
- Brute-force: O(N × M)

### Real-World Performance

Cholula, Mexico (5,097 nodes, 11,154 segments):
- 1,000 assignments: 0.05 seconds
- GPU upload (one-time): 0.2 seconds
- Interactive rendering: 60 FPS

---

## Architecture

### Technology Stack

Core:
- CGAL 6.0: Computational geometry
- pybind11: Python-C++ bindings
- NumPy/Pandas: Data processing

Geospatial:
- OSMnx: OpenStreetMap data access
- GeoPandas: Spatial data handling

Visualization:
- VisPy: GPU-accelerated rendering
- PyQt5: Window management

### Project Structure

```
acj/
├── __init__.py          # Public API
├── io.py                # Data loading
├── map_index.py         # Spatial indexing
├── graph.py             # Graph utilities
├── render.py            # Visualization
├── core/
│   └── src/
│       └── acj_core.cpp # CGAL bindings
└── tests/
    └── test_acj.py      # Test suite

examples/
├── example_acj.py       # Basic usage
└── example_realtime.py  # Interactive demo

CMakeLists.txt           # Build configuration
Dockerfile               # Container definition
Makefile                 # Build automation
```

---

## Development

### Running Tests

```bash
# All tests
make test

# Specific test class
docker run --user $(id -u):$(id -g) -v $(pwd):/workspace ubuntu-acj:1 \
  sh -c "PYTHONPATH=/workspace/build pytest acj/tests/ -v -k TestMapIndex"
```

Test coverage:
- Graph data validation
- OSMnx integration
- CGAL spatial queries
- Assignment correctness
- Data preservation

---

### Build Workflow

```bash
# Rebuild after changes
make clean
make build

# Interactive debugging
make shell-user

# Inside container
cd build && cmake .. && make
PYTHONPATH=/workspace/build python3
>>> import acj
>>> graph = acj.load_map("Liechtenstein")
```

---

## Known Limitations

1. Segment assignment not implemented: `assign_to_segments()` requires CGAL AABB tree implementation
2. Graph simplification pending: `simplify_graph()` currently returns input unchanged
3. Coordinates must be projected: Latitude/longitude must be converted to metric system (UTM)

---

## Planned Features

- Point-to-segment assignment using AABB tree
- Graph simplification with node merging
- Export visualizations to image/video
- Temporal analysis support
- Network-based distance calculation

---

## Troubleshooting

### Cannot connect to X server

Enable X11 forwarding:
```bash
xhost +local:docker
make example-realtime
```

### Module 'acj_core' not found

Recompile C++ extension:
```bash
make clean
make build
```

### City name not found

- Verify internet connection
- Check city name spelling (Nominatim format)
- Try broader query (country name)

---

## Citation

```bibtex
@software{acj2024,
  title={ACJ: Advanced Crime-to-Junction Assignment Library},
  author={Based on CGAL integration by Alejandro},
  year={2024},
  url={https://github.com/yourusername/acj}
}
```

---

## License

MIT License

---

## Contributing

Issues and pull requests welcome. Please include tests with new features.

**Acknowledgments:**
- Alejandro: CGAL integration and performance optimization
- CGAL Project: Computational geometry library
- OSMnx: OpenStreetMap data interface
- VisPy: GPU-accelerated visualization framework
