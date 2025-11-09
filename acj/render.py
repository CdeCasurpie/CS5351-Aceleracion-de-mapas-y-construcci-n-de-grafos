"""
Real-time GPU-accelerated visualization using VisPy.

This module provides a professional, interactive visualization tool for graph data
and analysis results using pure VisPy with OpenGL acceleration.
"""

from .io import GraphData
from .map_index import MapIndex
import pandas as pd
from typing import Optional
import numpy as np

try:
    from vispy import scene
    from vispy.scene import visuals
    import vispy.app
    from vispy.scene import Grid
except ImportError:
    raise ImportError(
        "VisPy is required for real-time visualization. "
        "Install with: pip install vispy PyQt5"
    )


class ACJVisualizer:
    """
    Real-time interactive analysis tool for street networks and heatmaps.
    
    This class creates a GPU-accelerated window with an on-screen display (OSD)
    for real-time feedback and keyboard controls for layer management.
    """
    
    def __init__(self, map_index: MapIndex, assignments: Optional[pd.DataFrame] = None,
                 title: str = "ACJ Real-Time Analysis Tool", canvas_size=(1400, 1000)):
        
        # --- 1. Preparación de Datos ---
        print("Preparing render data for GPU upload...")
        self.map_index = map_index
        self.assignments = assignments
        # NOTE: Using outlier_percentile from the last update for robust coloring
        self.render_data = map_index.get_render_data(
            assignments, 
            neighbor_influence=0.5, 
            outlier_percentile=98.0
        )
        
        # --- 2. Configuración del Canvas y la Vista ---
        self.canvas = scene.SceneCanvas(
            keys='interactive', title=title, size=canvas_size, show=True, bgcolor='#1e1e1e'
        )
        self.view = self.canvas.central_widget.add_view()
        
        # --- 3. Creación de los Elementos Visuales (en la GPU) ---
        self._create_visuals()
        
        # --- 4. Configuración de la Interfaz y Controles ---
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1.0
        
        self._set_initial_camera_view()
        
        self.visibility_state = {'nodes': True, 'segments': True, 'grid': True}
        self.mouse_coords = (0, 0)
        self._update_debugger_text()
        
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.view.camera.events.transform_change.connect(self._on_camera_change)
        
        print("GPU upload complete. Tool is ready for real-time interaction.")

    def _set_initial_camera_view(self):
        """Calcula el bounding box del mapa y centra la cámara."""
        node_vertices = self.render_data['node_vertices']
        if len(node_vertices) == 0:
            return

        x_min, y_min = node_vertices.min(axis=0)
        x_max, y_max = node_vertices.max(axis=0)

        padding_x = (x_max - x_min) * 0.05
        padding_y = (y_max - y_min) * 0.05
        
        self.view.camera.set_range(
            x=(x_min - padding_x, x_max + padding_x),
            y=(y_min - padding_y, y_max + padding_y),
            margin=0
        )

    def _create_visuals(self):
        """Crea y configura todos los objetos visuales de VisPy."""
        self.grid = visuals.GridLines(parent=self.view.scene, color=(0.5, 0.5, 0.5, 0.3))
        self.axis = visuals.XYZAxis(parent=self.view.scene)

        self.segments_visual = visuals.Line(
            pos=self.render_data['segment_vertices'], 
            color=self.render_data['segment_colors'],
            connect=self.render_data['segment_connectivity'], 
            width=5.0,
            method='gl', 
            parent=self.view.scene
        )
        
        self.nodes_visual = visuals.Markers(
            pos=self.render_data['node_vertices'], face_color=self.render_data['node_colors'],
            size=12, edge_width=0.5, edge_color='black', parent=self.view.scene
        )
        
        self.debugger_text = visuals.Text(
            "", pos=(15, 15), anchor_x='left', anchor_y='bottom',
            color='white', font_size=10, parent=self.canvas.scene
        )

    def _update_debugger_text(self):
        """Actualiza el panel de información en pantalla."""
        # --- CAMBIO PRINCIPAL: OBTENER LOS CONTEOS Y AÑADIRLOS AL TEXTO ---
        n_nodes = len(self.map_index.graph_data.nodes)
        n_segments = len(self.map_index.graph_data.segments)
        
        vis_nodes = "ON" if self.visibility_state['nodes'] else "OFF"
        vis_segments = "ON" if self.visibility_state['segments'] else "OFF"
        vis_grid = "ON" if self.visibility_state['grid'] else "OFF"
        
        zoom_level = self.view.camera.rect.width
        
        text = (
            f"--- ACJ ANALYSIS TOOL ---\n"
            f"Nodes: {n_nodes} | Segments: {n_segments}\n"  # <-- LÍNEA AÑADIDA
            f"Mouse Coords: ({self.mouse_coords[0]:.2f}, {self.mouse_coords[1]:.2f})\n"
            f"Zoom Level (Area Width): {zoom_level:.2f}\n"
            f"\n"
            f"--- LAYERS ---\n"
            f" (N) Nodes:    {vis_nodes}\n"
            f" (L) Segments: {vis_segments}\n"
            f" (G) Grid:     {vis_grid}\n"
            f"\n"
            f"--- CONTROLS ---\n"
            f" (R) Reset View | (Q) Quit"
        )
        self.debugger_text.text = text

    def _on_key_press(self, event):
        """Maneja las pulsaciones de teclas para los controles."""
        if event.key == 'n' or event.key == 'N':
            self.visibility_state['nodes'] = not self.visibility_state['nodes']
            self.nodes_visual.visible = self.visibility_state['nodes']
        elif event.key == 'l' or event.key == 'L':
            self.visibility_state['segments'] = not self.visibility_state['segments']
            # --- BUG FIX: AÑADIR ESTA LÍNEA PARA QUE EL CAMBIO SEA VISIBLE ---
            self.segments_visual.visible = self.visibility_state['segments']
        elif event.key == 'g' or event.key == 'G':
            self.visibility_state['grid'] = not self.visibility_state['grid']
            self.grid.visible = self.visibility_state['grid']
        elif event.key == 'r' or event.key == 'R':
            self._set_initial_camera_view()
        elif event.key == 'q' or event.key == 'Q' or event.key == 'Escape':
            self.canvas.close()
            
        self._update_debugger_text()

    def _on_mouse_move(self, event):
        """Actualiza las coordenadas del mouse en el debugger."""
        transform = self.view.camera.transform
        self.mouse_coords = transform.imap(event.pos)

    def _on_camera_change(self, event):
        """Actualiza el nivel de zoom cuando la cámara cambia."""
        self._update_debugger_text()
        
    def run(self):
        """Inicia el bucle de la aplicación interactiva."""
        print("\nStarting real-time interactive tool...")
        vispy.app.run()
        print("Visualizer closed.")




class ACJComparator:
    """Real-time visual comparison tool for two street networks."""

    def __init__(self, map_index_left: MapIndex, map_index_right: MapIndex,
                 assignments_left: Optional[pd.DataFrame] = None,
                 assignments_right: Optional[pd.DataFrame] = None,
                 title: str = "ACJ Graph Comparison",
                 title_left: str = "Original",
                 title_right: str = "Simplified",
                 canvas_size=(1600, 900)):

        print("Preparing render data for comparison...")
        self.render_data_left = map_index_left.get_render_data(assignments_left, outlier_percentile=98.0)
        self.render_data_right = map_index_right.get_render_data(assignments_right, outlier_percentile=98.0)

        self.canvas = scene.SceneCanvas(
            keys='interactive', title=title, size=canvas_size, show=True, bgcolor='#1e1e1e'
        )

        self.grid = self.canvas.central_widget.add_grid(margin=10)
        print("Creating shared PanZoomCamera...")
        self.camera = scene.PanZoomCamera(aspect=1.0)

        # Left and right linked views
        self.view_left = self.grid.add_view(row=0, col=0, border_color='gray', camera=self.camera)
        self.view_right = self.grid.add_view(row=0, col=1, border_color='gray', camera=self.camera)

        # Render left view
        self.segments_left = visuals.Line(
            pos=self.render_data_left['segment_vertices'], 
            color=self.render_data_left['segment_colors'],
            connect=self.render_data_left['segment_connectivity'], 
            width=5.0, method='gl', parent=self.view_left.scene
        )
        self.nodes_left = visuals.Markers(
            pos=self.render_data_left['node_vertices'],
            face_color=self.render_data_left['node_colors'],
            size=12, edge_width=0.5, edge_color='yellow',
            parent=self.view_left.scene
        )

        # Render right view
        self.segments_right = visuals.Line(
            pos=self.render_data_right['segment_vertices'], 
            color=self.render_data_right['segment_colors'],
            connect=self.render_data_right['segment_connectivity'], 
            width=5.0, method='gl', parent=self.view_right.scene
        )
        self.nodes_right = visuals.Markers(
            pos=self.render_data_right['node_vertices'],
            face_color=self.render_data_right['node_colors'],
            size=12, edge_width=0.5, edge_color='yellow',
            parent=self.view_right.scene
        )

        # On-screen display (OSD) text at bottom
        self.debugger_text_left = visuals.Text("", pos=(15, 15), anchor_x='left', anchor_y='bottom',
                                               color='white', font_size=10, parent=self.canvas.scene)
        self.debugger_text_right = visuals.Text("", pos=(canvas_size[0] - 15, 15),
                                                anchor_x='right', anchor_y='bottom',
                                                color='white', font_size=10, parent=self.canvas.scene)
        
        # Large descriptive titles at the top with labels
        center_left = canvas_size[0] // 4
        center_right = 3 * canvas_size[0] // 4
        top_pos = canvas_size[1] - 25
        
        # Left side label and title
        self.label_text_left = visuals.Text("LEFT", pos=(center_left, top_pos + 25),
                                            anchor_x='center', anchor_y='top', 
                                            color='#00ff00', font_size=16, bold=True, 
                                            parent=self.canvas.scene)
        self.title_text_left = visuals.Text(title_left, pos=(center_left, top_pos),
                                            anchor_x='center', anchor_y='top', 
                                            color='white', font_size=16, bold=True, 
                                            parent=self.canvas.scene)
        
        # Right side label and title
        self.label_text_right = visuals.Text("RIGHT", pos=(center_right, top_pos + 25),
                                             anchor_x='center', anchor_y='top', 
                                             color='#ff8800', font_size=16, bold=True, 
                                             parent=self.canvas.scene)
        self.title_text_right = visuals.Text(title_right, pos=(center_right, top_pos),
                                             anchor_x='center', anchor_y='top', 
                                             color='white', font_size=16, bold=True, 
                                             parent=self.canvas.scene)

        self._set_initial_camera_view()
        self.visibility_state = {'nodes': True, 'segments': True}

        # Event bindings
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.resize.connect(self._on_canvas_resize)
        self.view_left.camera.events.transform_change.connect(self._on_camera_change)
        self._update_debugger_text()

        print("Comparison tool ready. Keys: (N)odes, (L)ines, (R)eset, (Q)uit")

    def _set_initial_camera_view(self):
        """Center camera around the left view’s bounding box."""
        node_vertices = self.render_data_left['node_vertices']
        if len(node_vertices) == 0:
            return
        x_min, y_min = node_vertices.min(axis=0)
        x_max, y_max = node_vertices.max(axis=0)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        self.view_left.camera.set_range(
            x=(x_min - padding_x, x_max + padding_x),
            y=(y_min - padding_y, y_max + padding_y),
            margin=0
        )

    def _update_debugger_text(self):
        """Update on-screen debug info for both views."""
        vis_nodes = "ON" if self.visibility_state['nodes'] else "OFF"
        vis_segments = "ON" if self.visibility_state['segments'] else "OFF"
        zoom_level = self.view_left.camera.rect.width

        common_text = (
            f"Zoom: {zoom_level:.2f}\n"
            f"(N) Nodes: {vis_nodes}\n"
            f"(L) Segments: {vis_segments}\n"
            f"(R) Reset | (Q) Quit"
        )
        self.debugger_text_left.text = (
            f"Nodes: {len(self.render_data_left['node_vertices'])}\n"
            f"Segments: {len(self.render_data_left['segment_connectivity'])}\n"
            f"{common_text}"
        )
        self.debugger_text_right.text = (
            f"Nodes: {len(self.render_data_right['node_vertices'])}\n"
            f"Segments: {len(self.render_data_right['segment_connectivity'])}\n"
            f"{common_text}"
        )
        
        # Update positions dynamically based on current canvas size
        canvas_width = self.canvas.size[0]
        canvas_height = self.canvas.size[1]
        
        self.debugger_text_right.pos = (canvas_width - 15, 15)
        
        # Update title positions to stay centered
        center_left = canvas_width // 4
        center_right = 3 * canvas_width // 4
        top_pos = canvas_height - 25
        
        self.label_text_left.pos = (center_left, top_pos + 25)
        self.title_text_left.pos = (center_left, top_pos)
        self.label_text_right.pos = (center_right, top_pos + 25)
        self.title_text_right.pos = (center_right, top_pos)

    def _on_key_press(self, event):
        """Handle keypress events for toggling or resetting view."""
        if event.key in ('n', 'N'):
            self.visibility_state['nodes'] = not self.visibility_state['nodes']
            self.nodes_left.visible = self.nodes_right.visible = self.visibility_state['nodes']
        elif event.key in ('l', 'L'):
            self.visibility_state['segments'] = not self.visibility_state['segments']
            self.segments_left.visible = self.segments_right.visible = self.visibility_state['segments']
        elif event.key in ('r', 'R'):
            self._set_initial_camera_view()
        elif event.key in ('q', 'Q', 'Escape'):
            self.canvas.close()
        self._update_debugger_text()

    def _on_camera_change(self, event):
        """Refresh debug info when camera zoom or pan changes."""
        self._update_debugger_text()
    
    def _on_canvas_resize(self, event):
        """Update positions when canvas is resized."""
        self._update_debugger_text()

    def run(self):
        """Start the interactive visualization loop."""
        print("\nStarting comparison visualizer...")
        vispy.app.run()
        print("Visualizer closed.")




# --- Public Visualization Functions ---

def render_realtime(map_index: MapIndex, assignments: Optional[pd.DataFrame] = None,
                   title: str = "Street Network Visualization"):
    """Render an interactive real-time visualization of a single street network."""
    visualizer = ACJVisualizer(map_index, assignments, title=title)
    visualizer.run()
    return visualizer

def render_graph(map_index: MapIndex, title: str = "Street Network Graph"):
    """Render the base street network without any assignments."""
    return render_realtime(map_index, assignments=None, title=title)

def render_heatmap(map_index: MapIndex, assignments: pd.DataFrame,
                  title: str = "Crime Density Heatmap"):
    """Render a heatmap visualization using the given assignments."""
    return render_realtime(map_index, assignments, title=title)

def render_comparison(map_index_left: MapIndex, map_index_right: MapIndex,
                      assignments_left: Optional[pd.DataFrame] = None,
                      assignments_right: Optional[pd.DataFrame] = None,
                      title: str = "ACJ Graph Comparison",
                      title_left: str = "Original Graph",
                      title_right: str = "Simplified Graph"):
    """Render a side-by-side comparison between two MapIndex objects."""
    visualizer = ACJComparator(
        map_index_left, map_index_right,
        assignments_left, assignments_right,
        title, title_left, title_right
    )
    visualizer.run()
    return visualizer

