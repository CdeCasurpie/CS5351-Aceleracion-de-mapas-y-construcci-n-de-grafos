"""
Real-time GPU-accelerated visualization using VisPy.

This module provides interactive visualization of graph data and analysis results
using VisPy with OpenGL acceleration. Unlike matplotlib, this renders in real-time
with smooth zoom, pan, and rotation capabilities.
"""

from .io import GraphData
from .map_index import MapIndex
import pandas as pd
import numpy as np
from typing import Optional

try:
    from vispy import scene
    from vispy.scene import visuals
    import vispy.app
except ImportError:
    raise ImportError(
        "VisPy is required for real-time visualization. "
        "Install with: pip install vispy PyQt5"
    )


class ACJVisualizer:
    """
    Real-time interactive visualizer for street networks and crime heatmaps.
    
    This class creates a GPU-accelerated window using VisPy that allows
    real-time interaction (zoom, pan) with large datasets. All data is
    uploaded to the GPU once during initialization for maximum performance.
    
    Controls:
        - Mouse drag: Pan the view
        - Mouse wheel: Zoom in/out
        - Right mouse drag: Rotate (3D mode only)
        - 'N': Toggle nodes visibility
        - 'R': Reset camera view
        - 'Q' or ESC: Close window
    
    Example:
        >>> graph = acj.load_map("Cholula, Puebla, Mexico")
        >>> map_index = acj.MapIndex(graph)
        >>> assignments = map_index.assign_to_endpoints(crimes)
        >>> visualizer = acj.ACJVisualizer(map_index, assignments)
        >>> visualizer.run()
    """
    
    def __init__(self, map_index: MapIndex, assignments: Optional[pd.DataFrame] = None,
                 title: str = "ACJ Real-Time Visualizer", canvas_size=(1200, 900),
                 show_nodes: bool = False, show_segments: bool = True):
        """
        Initialize the real-time visualizer.
        
        Args:
            map_index: MapIndex object containing the graph data
            assignments: Optional DataFrame with crime assignments for heatmap coloring
            title: Window title
            canvas_size: Window size (width, height) in pixels
            show_nodes: Whether to render nodes initially (default: False, toggle with 'N')
            show_segments: Whether to render segments
        
        Notes:
            This constructor uploads all data to GPU. Rendering happens in run().
        """
        self.map_index = map_index
        self.assignments = assignments
        self.title = title
        self.nodes_visible = show_nodes
        
        # Pre-compute all render data (CPU-side calculation, done once)
        print("Preparing render data for GPU upload...")
        self.render_data = map_index.get_render_data(assignments)
        
        # Create the canvas (window)
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            title=title,
            size=canvas_size,
            show=True,
            bgcolor='white'
        )
        
        # Create the view (camera viewport)
        self.view = self.canvas.central_widget.add_view()
        
        # Upload segments to GPU with subtle outline for visibility
        if show_segments:
            print(f"Uploading {len(self.render_data['segment_connectivity'])} segments to GPU...")
            
            # First, draw black outlines (just slightly wider)
            self.segments_outline = visuals.Line(
                pos=self.render_data['segment_vertices'],
                color='gray',
                connect=self.render_data['segment_connectivity'],
                width=0.5,  # Just slightly wider than colored lines
                method='gl',
                parent=self.view.scene
            )
            
            # Then, draw colored segments with gradient on top
            self.segments_visual = visuals.Line(
                pos=self.render_data['segment_vertices'],
                color=self.render_data['segment_colors'],
                connect=self.render_data['segment_connectivity'],
                width=6.0,  # Main line width - the gradient is here
                method='gl',
                parent=self.view.scene
            )
        
        # Upload nodes to GPU with heatmap colors (no blue border)
        print(f"Uploading {len(self.render_data['node_vertices'])} nodes to GPU...")
        self.nodes_visual = visuals.Markers(
            pos=self.render_data['node_vertices'],
            face_color=self.render_data['node_colors'],
            size=17,
            edge_width=0,  # No border
            parent=self.view.scene
        )
        
        # Set initial visibility
        self.nodes_visual.visible = self.nodes_visible
        
        # Configure camera for 2D pan/zoom
        self.view.camera = 'panzoom'
        self.view.camera.set_range()
        
        # Add statistics text overlay if assignments provided
        if assignments is not None:
            self._add_statistics_overlay()
        
        # Connect keyboard events
        self.canvas.events.key_press.connect(self._on_key_press)
        
        print("GPU upload complete. Ready for real-time rendering.")
    
    def _add_statistics_overlay(self):
        """Add text overlay with statistics."""
        total_crimes = len(self.assignments)
        crime_counts = self.assignments['assigned_node_id'].value_counts()
        nodes_with_crimes = len(crime_counts)
        total_nodes = len(self.map_index.graph_data.nodes)
        max_crimes = crime_counts.max() if len(crime_counts) > 0 else 0
        
        nodes_status = "Hidden" if not self.nodes_visible else "Visible"
        
        stats_text = (
            f"Total Crimes: {total_crimes}\n"
            f"Affected Nodes: {nodes_with_crimes}/{total_nodes}\n"
            f"Max Crimes/Node: {max_crimes}\n"
            f"Nodes: {nodes_status}\n"
            f"\n"
            f"Controls:\n"
            f"  Mouse: Pan\n"
            f"  Wheel: Zoom\n"
            f"  N: Toggle Nodes\n"
            f"  R: Reset view\n"
            f"  Q: Quit"
        )
        
        self.text_visual = visuals.Text(
            stats_text,
            pos=(20, 30),
            anchor_x='left',
            anchor_y='top',
            color='black',
            font_size=10,
            parent=self.view
        )
    
    def _update_statistics_text(self):
        """Update the statistics text overlay."""
        if hasattr(self, 'text_visual') and self.assignments is not None:
            total_crimes = len(self.assignments)
            crime_counts = self.assignments['assigned_node_id'].value_counts()
            nodes_with_crimes = len(crime_counts)
            total_nodes = len(self.map_index.graph_data.nodes)
            max_crimes = crime_counts.max() if len(crime_counts) > 0 else 0
            
            nodes_status = "Hidden" if not self.nodes_visible else "Visible"
            
            stats_text = (
                f"Total Crimes: {total_crimes}\n"
                f"Affected Nodes: {nodes_with_crimes}/{total_nodes}\n"
                f"Max Crimes/Node: {max_crimes}\n"
                f"Nodes: {nodes_status}\n"
                f"\n"
                f"Controls:\n"
                f"  Mouse: Pan\n"
                f"  Wheel: Zoom\n"
                f"  N: Toggle Nodes\n"
                f"  R: Reset view\n"
                f"  Q: Quit"
            )
            
            self.text_visual.text = stats_text
    
    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'n' or event.key == 'N':
            # Toggle nodes visibility
            self.nodes_visible = not self.nodes_visible
            self.nodes_visual.visible = self.nodes_visible
            status = "visible" if self.nodes_visible else "hidden"
            print(f"Nodes: {status}")
            self._update_statistics_text()
            
        elif event.key == 'r' or event.key == 'R':
            # Reset camera view
            self.view.camera.set_range()
            print("Camera view reset")
            
        elif event.key == 'q' or event.key == 'Q' or event.key == 'Escape':
            # Close window
            self.canvas.close()
    
    def run(self):
        """
        Start the interactive visualization loop.
        
        This blocks until the window is closed. All rendering is handled
        by VisPy/OpenGL automatically with GPU acceleration.
        """
        print(f"\nStarting real-time visualization...")
        print(f"Window: {self.title}")
        print("Use mouse to pan/zoom.")
        print("Press 'N' to toggle nodes, 'Q' to quit.\n")
        
        # Start the VisPy application event loop
        vispy.app.run()


def render_realtime(map_index: MapIndex, assignments: Optional[pd.DataFrame] = None,
                   title: str = "Street Network Visualization") -> ACJVisualizer:
    """
    Launch real-time interactive visualization.
    
    This is the main convenience function for users. It creates an ACJVisualizer
    and immediately starts the rendering loop.
    
    Args:
        map_index: MapIndex object with the graph data
        assignments: Optional crime assignments for heatmap coloring
        title: Window title
    
    Returns:
        ACJVisualizer instance (mostly for advanced users)
    
    Example:
        >>> graph = acj.load_map("Cholula, Puebla, Mexico")
        >>> map_index = acj.MapIndex(graph)
        >>> assignments = map_index.assign_to_endpoints(crimes)
        >>> acj.render_realtime(map_index, assignments)
    
    Notes:
        This function blocks until the window is closed.
    """
    visualizer = ACJVisualizer(map_index, assignments, title=title)
    visualizer.run()
    return visualizer


def render_heatmap(map_index: MapIndex, assignments: pd.DataFrame,
                  title: str = "Crime Density Heatmap") -> ACJVisualizer:
    """
    Launch real-time interactive heatmap visualization.
    
    This is a convenience wrapper around render_realtime() with a specific title
    for heatmap visualizations.
    
    Args:
        map_index: MapIndex object with the graph data
        assignments: Crime assignments DataFrame (required for heatmap)
        title: Window title
    
    Returns:
        ACJVisualizer instance
    
    Example:
        >>> acj.render_heatmap(map_index, assignments)
    """
    return render_realtime(map_index, assignments, title=title)


def render_graph(map_index: MapIndex, title: str = "Street Network Graph") -> ACJVisualizer:
    """
    Launch real-time visualization of basic street network.
    
    This shows the graph without any crime heatmap coloring.
    
    Args:
        map_index: MapIndex object with the graph data
        title: Window title
    
    Returns:
        ACJVisualizer instance
    
    Example:
        >>> graph = acj.load_map("Cholula, Puebla, Mexico")
        >>> map_index = acj.MapIndex(graph)
        >>> acj.render_graph(map_index)
    """
    return render_realtime(map_index, assignments=None, title=title)
