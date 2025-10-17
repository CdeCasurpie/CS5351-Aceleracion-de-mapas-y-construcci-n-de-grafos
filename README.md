# ACJ - Advanced Geospatial Analysis Library

A high-performance Python library for efficient point-to-graph assignment using CGAL (Computational Geometry Algorithms Library). Designed for large-scale urban analytics, crime mapping, and network analysis.

## Overview

ACJ provides fast spatial indexing and querying capabilities for assigning points (e.g., crime locations, customer addresses) to graph elements (nodes and segments of street networks). The library wraps CGAL's spatial data structures with a convenient Python interface.

**Key Features:**
- Fast point-to-node assignment using CGAL Delaunay triangulation (O(N log M) complexity)
- Support for custom graph data via pandas DataFrames
- Standardized data format for interoperability
- Docker-based development environment for reproducibility
- Comprehensive test suite

**Based on work by:** Alejandro (CGAL integration and performance optimization)

---

## Installation

### Prerequisites

- Docker (recommended for reproducible builds)
- OR: Linux with CGAL, pybind11, Python 3.8+, CMake 3.15+

### Quick Start with Docker

```bash
# Build the Docker image with all dependencies
make build

# Run tests
make test-acj

# Run example
make example-acj
```

### Manual Installation (without Docker)

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake python3-dev python3-pip
sudo apt-get install libcgal-dev pybind11-dev

# Install Python dependencies
pip install -r requirements.txt

# Compile the C++ extension
mkdir build && cd build
cmake .. && make -j$(nproc)

# Add build directory to PYTHONPATH
export PYTHONPATH=/path/to/pylib/build:$PYTHONPATH
```

---

## Usage

### Basic Example

```python
import acj
import pandas as pd

# Step 1: Load graph data (nodes and segments)
nodes = pd.DataFrame({
    'node_id': [0, 1, 2, 3],
    'x': [0.0, 100.0, 200.0, 100.0],
    'y': [0.0, 0.0, 100.0, 100.0]
})

segments = pd.DataFrame({
    'segment_id': [0, 1, 2],
    'node_start': [0, 1, 2],
    'node_end': [1, 2, 3],
    'x1': [0.0, 100.0, 200.0],
    'y1': [0.0, 0.0, 100.0],
    'x2': [100.0, 200.0, 100.0],
    'y2': [0.0, 100.0, 100.0]
})

graph = acj.load_graph(nodes, segments)

# Step 2: Create spatial index
map_index = acj.MapIndex(graph)

# Step 3: Assign points to nearest nodes
crimes = pd.DataFrame({
    'point_id': [0, 1, 2],
    'x': [50.0, 150.0, 25.0],
    'y': [5.0, 10.0, 90.0],
    'crime_type': ['robbery', 'assault', 'theft']
})

assignments = map_index.assign_to_endpoints(crimes)
print(assignments[['point_id', 'assigned_node_id', 'distance']])
```

---

## Data Format Standards

ACJ uses standardized pandas DataFrame formats for all inputs.

### Nodes DataFrame

Required columns:
- `node_id` (int): Unique identifier for each node
- `x` (float): X coordinate in projected system (e.g., UTM meters)
- `y` (float): Y coordinate in projected system (e.g., UTM meters)

Optional columns: Any additional metadata (preserved in outputs)

### Segments DataFrame

Required columns:
- `segment_id` (int): Unique identifier for each segment
- `node_start` (int): Reference to starting node_id
- `node_end` (int): Reference to ending node_id
- `x1`, `y1` (float): Coordinates of start point
- `x2`, `y2` (float): Coordinates of end point

Optional columns: Any additional metadata (length, speed limit, etc.)

### Points DataFrame

Required columns:
- `point_id` (int): Unique identifier for each point
- `x` (float): X coordinate
- `y` (float): Y coordinate

Optional columns: Any additional data (crime type, timestamp, etc.) - preserved in output

---

## API Reference

### Core Functions

#### `acj.load_graph(nodes_df, segments_df)`
Load graph data from pandas DataFrames.

**Parameters:**
- `nodes_df`: DataFrame with node data
- `segments_df`: DataFrame with segment data

**Returns:** `GraphData` object

---

#### `acj.MapIndex(graph_data)`
Create spatial index for efficient queries.

**Parameters:**
- `graph_data`: GraphData object from `load_graph()`

**Methods:**
- `assign_to_endpoints(points_df)`: Assign points to nearest nodes
- `assign_to_segments(points_df)`: [PENDING] Assign points to nearest segments

---

### Pending Features

The following functions are planned but not yet implemented:

#### `acj.load_map(city_name, cache_dir="./cache")`
Load street network from OpenStreetMap using OSMnx.

**Status:** PENDING IMPLEMENTATION

---

#### `acj.simplify_graph(graph_data, threshold_meters=10.0)`
Simplify graph by merging nearby nodes.

**Status:** Currently returns input unchanged (implementation pending)

---

#### `acj.render_graph(graph_data, **kwargs)`
Visualize street network graph.

**Status:** PENDING IMPLEMENTATION

---

#### `acj.render_heatmap(graph_data, assignments, **kwargs)`
Create heatmap visualization of assignment results.

**Status:** PENDING IMPLEMENTATION

---

## Performance

ACJ uses CGAL's Delaunay triangulation for efficient spatial queries:

- **Complexity:** O(M log M) index construction + O(N log M) queries
- **Benchmark results** (100k query points → 50k target nodes):
  - CGAL: 1.29 seconds
  - Brute-force: 7.89 seconds
  - **Speedup: 6.1x**

See `benchmark.py` for detailed performance comparisons.

---

## Development

### Project Structure

```
pylib/
├── acj/                    # Main Python package
│   ├── __init__.py         # Package initialization
│   ├── io.py               # Data loading functions
│   ├── map_index.py        # MapIndex class (core API)
│   ├── graph.py            # Graph utilities
│   ├── render.py           # Visualization functions
│   ├── core/               # C++ extension module
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       └── acj_core.cpp  # CGAL wrapper
│   └── tests/
│       └── test_acj.py     # Test suite
├── examples/
│   └── example_acj.py      # Usage examples
├── CMakeLists.txt          # Main build configuration
├── Makefile                # Build automation
├── Dockerfile              # Docker environment
└── README.md               # This file
```

### Running Tests

```bash
# All ACJ tests
make test-acj

# Specific test file
docker run --user $(id -u):$(id -g) -v $(pwd):/workspace ubuntu-acj:1 \
  sh -c "cd /workspace && PYTHONPATH=/workspace/build python3 -m pytest acj/tests/test_acj.py::TestMapIndex -v"

# Legacy matcher tests (from Alejandro's work)
make test
```

### Adding New Features

1. Add Python interface in appropriate module (`io.py`, `map_index.py`, etc.)
2. If C++ implementation needed, add to `acj/core/src/acj_core.cpp`
3. Update `CMakeLists.txt` if new dependencies required
4. Add tests to `acj/tests/test_acj.py`
5. Update this README

---

## Known Limitations

1. **Segment assignment not implemented:** `assign_to_segments()` requires AABB tree support
2. **OSMnx integration pending:** `load_map()` not yet implemented
3. **Visualization pending:** Rendering functions not yet implemented
4. **Graph simplification pending:** Currently returns unchanged graph

---

## Citation

If you use this library in your research, please cite:

```
ACJ - Advanced Geospatial Analysis Library
Based on CGAL integration work by Alejandro
https://github.com/yourusername/acj
```

---

## License

MIT License (see LICENSE file)

---

## Contact

For questions or contributions, please open an issue on GitHub.
