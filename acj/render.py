"""
Real-time GPU-accelerated visualization using VisPy.

This module provides a professional, interactive visualization tool for graph data
and analysis results using pure VisPy with OpenGL acceleration.
"""

from .io import GraphData
from .map_index import MapIndex
import pandas as pd
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
        self.render_data = map_index.get_render_data(assignments)
        
        # --- 2. Configuración del Canvas y la Vista ---
        self.canvas = scene.SceneCanvas(
            keys='interactive', title=title, size=canvas_size, show=True, bgcolor='#1e1e1e'
        )
        self.view = self.canvas.central_widget.add_view()
        
        # --- 3. Creación de los Elementos Visuales (en la GPU) ---
        self._create_visuals()
        
        # --- 4. Configuración de la Interfaz y Controles ---
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1.0 # Mantiene la proporción correcta
        
        # --- FIX: CÁLCULO MANUAL DEL ZOOM Y CENTRADO INICIAL ---
        # Calculamos los límites del mapa y establecemos la vista de la cámara
        # para que todo el mapa sea visible al inicio.
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

        # Añadir un 5% de padding para que no toque los bordes
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
            width=3.0,
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
        # ... (esta función no necesita cambios)
        vis_nodes = "ON" if self.visibility_state['nodes'] else "OFF"
        vis_segments = "ON" if self.visibility_state['segments'] else "OFF"
        vis_grid = "ON" if self.visibility_state['grid'] else "OFF"
        
        zoom_level = self.view.camera.rect.width
        
        text = (
            f"--- ACJ ANALYSIS TOOL ---\n"
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
        # ... (esta función no necesita cambios)
        if event.key == 'n' or event.key == 'N':
            self.visibility_state['nodes'] = not self.visibility_state['nodes']
            self.nodes_visual.visible = self.visibility_state['nodes']
        elif event.key == 'l' or event.key == 'L':
            self.visibility_state['segments'] = not self.visibility_state['segments']
        elif event.key == 'g' or event.key == 'G':
            self.visibility_state['grid'] = not self.visibility_state['grid']
            self.grid.visible = self.visibility_state['grid']
        elif event.key == 'r' or event.key == 'R':
            self._set_initial_camera_view() # Usamos nuestra función de reseteo
        elif event.key == 'q' or event.key == 'Q' or event.key == 'Escape':
            self.canvas.close()
            
        self._update_debugger_text()

    def _on_mouse_move(self, event):
        """Actualiza las coordenadas del mouse en el debugger."""
        # ... (esta función no necesita cambios)
        transform = self.view.camera.transform
        self.mouse_coords = transform.imap(event.pos)

    def _on_camera_change(self, event):
        """Actualiza el nivel de zoom cuando la cámara cambia."""
        # ... (esta función no necesita cambios)
        self._update_debugger_text()
        
    def run(self):
        """Inicia el bucle de la aplicación interactiva."""
        print("\nStarting real-time interactive tool...")
        vispy.app.run()
        print("Visualizer closed.")

# --- Funciones de Conveniencia para el Usuario Final ---
def render_realtime(map_index: MapIndex, assignments: Optional[pd.DataFrame] = None,
                   title: str = "Street Network Visualization"):
    visualizer = ACJVisualizer(map_index, assignments, title=title)
    visualizer.run()
    return visualizer

def render_heatmap(map_index: MapIndex, assignments: pd.DataFrame,
                  title: str = "Crime Density Heatmap"):
    return render_realtime(map_index, assignments, title=title)

def render_graph(map_index: MapIndex, title: str = "Street Network Graph"):
    return render_realtime(map_index, assignments=None, title=title)