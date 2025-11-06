"""
Real-time interactive visualization example with ACJ library.

This example demonstrates GPU-accelerated real-time visualization:
1. Loads a real city map from OpenStreetMap
2. Generates random crime points within the city bounds
3. Assigns crimes to nearest street nodes
4. Launches interactive VisPy window with real-time zoom/pan
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import numpy as np
import pandas as pd


def generate_random_points(
    graph_data,
    n_points=500,
    seed=42,
    num_hotspots=60,
    hotspot_radius=400.0
):
    """
    Generates random points in clusters around selected "hotspot" segments.

    This method simulates realistic "danger zones" by:
    1. Selecting a few random street segments as hotspot centers.
    2. Assigning a random number of points to each hotspot.
    3. Generating points in a Gaussian cluster around each hotspot's segment.

    Args:
        graph_data: GraphData object with the street network.
        n_points: Total number of random points to generate.
        seed: Random seed for reproducibility.
        num_hotspots: The number of "danger zones" to create.
        hotspot_radius: The standard deviation (in meters) for the point cluster
                        around each hotspot. Larger values create more spread.

    Returns:
        DataFrame with columns ['point_id', 'x', 'y', 'crime_type'].
    """
    np.random.seed(seed)

    segments_df = graph_data.segments
    if len(segments_df) == 0 or num_hotspots == 0 or n_points == 0:
        return pd.DataFrame({'point_id': [], 'x': [], 'y': [], 'crime_type': []})

    # 1. Select 'num_hotspots' unique segments to be the centers of our clusters
    # Ensure we don't select more hotspots than available segments
    actual_num_hotspots = min(num_hotspots, len(segments_df))
    hotspot_indices = np.random.choice(
        segments_df.index,
        size=actual_num_hotspots,
        replace=False
    )
    hotspot_segments = segments_df.loc[hotspot_indices]

    # 2. Assign a random number of points to each hotspot
    # Generate random weights and normalize them to sum to n_points
    random_weights = np.random.rand(actual_num_hotspots) + 0.1 # Add 0.1 to avoid zero-sized hotspots
    points_per_hotspot = (random_weights / random_weights.sum() * n_points).astype(int)

    # Adjust for rounding errors to ensure the sum is exactly n_points
    diff = n_points - points_per_hotspot.sum()
    points_per_hotspot[-1] += diff

    # 3. Generate points for each hotspot
    all_x = []
    all_y = []

    for i, num_points_in_hotspot in enumerate(points_per_hotspot):
        if num_points_in_hotspot == 0:
            continue

        segment = hotspot_segments.iloc[i]

        # Generate base points along the central segment line
        t = np.random.rand(num_points_in_hotspot)
        base_x = segment['x1'] + t * (segment['x2'] - segment['x1'])
        base_y = segment['y1'] + t * (segment['y2'] - segment['y1'])

        # Add a 2D Gaussian (normal) spread to create a cluster
        offsets = np.random.normal(0, hotspot_radius, size=(num_points_in_hotspot, 2))

        final_x = base_x + offsets[:, 0]
        final_y = base_y + offsets[:, 1]

        all_x.append(final_x)
        all_y.append(final_y)

    # Combine all points into final arrays
    x_coords = np.concatenate(all_x)
    y_coords = np.concatenate(all_y)

    # Generate random crime types for all points
    crime_types = np.random.choice(
        ['robbery', 'assault', 'theft', 'vandalism', 'burglary'],
        size=n_points
    )

    points_df = pd.DataFrame({
        'point_id': range(n_points),
        'x': x_coords,
        'y': y_coords,
        'crime_type': crime_types
    })

    return points_df

def main():
    print("=" * 80)
    print("ACJ Library - Real-Time Interactive Visualization Example")
    print("=" * 80)
    print()

    # Configuration
    city_name = "Barranco, Lima, Peru"
    n_crimes = 1000000

    # Step 1: Load city map from OpenStreetMap
    print(f"[1/4] Loading street network for '{city_name}'...")
    print("       (Using cached data if available)")
    print()

    try:
        graph = acj.load_map(city_name, cache_dir="./cache", network_type="drive")
    except Exception as e:
        print(f"ERROR: Could not load map: {e}")
        print("\nTrying alternative city: 'Belluno, Italy'...")
        city_name = "Belluno, Italy"
        try:
            graph = acj.load_map(city_name, cache_dir="./cache", network_type="drive")
        except Exception as e2:
            print(f"ERROR: Could not load alternative map: {e2}")
            print("\nPlease check your internet connection and OSMnx installation.")
            return

    print(f"Loaded: {len(graph.nodes)} nodes, {len(graph.segments)} segments")
    print()

    # Step 2: Generate random crime points
    print(f"[2/4] Generating {n_crimes} random crime points...")
    crimes = generate_random_points(graph, n_points=n_crimes, seed=random.randint(0, 10000))

    print(f"Generated {len(crimes)} crime points")
    print(f"Crime types: {crimes['crime_type'].value_counts().to_dict()}")
    print()

    # Step 3: Create spatial index and assign crimes
    print("[3/4] Building spatial index and assigning crimes...")
    map_index = acj.MapIndex(graph)
    assignments = map_index.assign_to_endpoints(crimes)

    avg_distance = assignments['distance'].mean()
    max_distance = assignments['distance'].max()
    print(f"Assignment complete!")
    print(f"  Average distance to nearest node: {avg_distance:.2f} meters")
    print(f"  Maximum distance to nearest node: {max_distance:.2f} meters")
    print()

    # Show top crime hotspots
    hotspots = assignments['assigned_node_id'].value_counts().head(5)
    print("Top 5 crime hotspots (node_id: count):")
    for node_id, count in hotspots.items():
        print(f"  Node {node_id}: {count} crimes")
    print()

    # Step 4: Launch real-time interactive visualization
    print("[4/4] Launching real-time interactive visualization...")
    print()
    print("=" * 80)
    print("INTERACTIVE WINDOW CONTROLS:")
    print("=" * 80)
    print("  Mouse drag        : Pan the view")
    print("  Mouse wheel       : Zoom in/out")
    print("  'R' key           : Reset camera view")
    print("  'Q' or ESC        : Close window")
    print("=" * 80)
    print()
    print("Opening VisPy window...")
    print("(All data is uploaded to GPU for smooth real-time interaction)")
    print()

    # Launch the real-time visualizer
    # This will block until the window is closed
    acj.render_heatmap(
        map_index,
        assignments,
        title=f"Crime Heatmap - {city_name} ({n_crimes} crimes)"
    )

    print()
    print("=" * 80)
    print("Visualization closed. Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
