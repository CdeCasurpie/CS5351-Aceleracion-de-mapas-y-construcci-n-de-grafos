"""
ALGORITMO DE SIMPLIFICACIÓN DE GRAFO: FUSIÓN DE VÉRTICES Y COLAPSO A SEGMENTOS

FASE 1: Clustering Vértice-Vértice (DSU)
   - Agrupa vértices cercanos (distancia < 2R) usando conjuntos disjuntos.
   - Colapsa clusters en centroides.

FASE 2: Colapso Vértice-Segmento (Proyección Ortogonal)
   - Detecta vértices aislados que están dentro del radio de influencia de un segmento (arista).
   - "Radio de influencia" implica que tanto el vértice como el segmento tienen grosor R.
   - Operación topológica: Proyectar vértice -> Cortar arista -> Recablear vecinos.

Visualización: Renderizado interactivo con Matplotlib y Sliders.
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from shapely.geometry import Point, LineString
import math

# --- Estructura de Datos: Disjoint Set ---
class DisjointSet:
    def __init__(self, elements=None):
        self.parent = {}
        if elements:
            for elem in elements:
                self.parent[elem] = elem

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2

# --- Geometría Computacional ---

def distancia_punto_segmento(px, py, x1, y1, x2, y2):
    """
    Calcula la distancia mínima entre un punto (px, py) y un segmento definido por
    (x1, y1) y (x2, y2). Retorna la distancia, el punto de proyección y el factor t.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0: # El segmento es un punto
        dist = math.hypot(px - x1, py - y1)
        return dist, x1, y1, 0.0

    # Proyección del vector AP sobre AB (producto punto normalizado)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    
    # Restringir t al segmento [0, 1]
    t = max(0, min(1, t))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, proj_x, proj_y, t

# --- Operaciones Topológicas ---

def colapsar_vertices_cluster(cluster_nodes, G):
    """Fase 1: Colapsa un grupo de vértices en su centroide."""
    if not cluster_nodes: return G

    # Coordenadas y Centroide
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in cluster_nodes if n in G]
    if not coords: return G
    
    avg_x = sum(c[0] for c in coords) / len(coords)
    avg_y = sum(c[1] for c in coords) / len(coords)
    
    # Nodo Representante (estable)
    new_id = cluster_nodes[0]
    
    # Recableado
    edges_to_add = []
    nodes_to_remove = [n for n in cluster_nodes if n != new_id]
    
    is_directed = G.is_directed()
    is_multigraph = G.is_multigraph()

    for n in cluster_nodes:
        if n not in G: continue
        
        # Salientes
        out_iter = G.out_edges(n, data=True, keys=True) if is_multigraph else G.out_edges(n, data=True)
        for edge in out_iter:
            target = edge[1]
            data = edge[-1]
            if target not in cluster_nodes:
                edges_to_add.append((new_id, target, data))

        # Entrantes
        if is_directed:
            in_iter = G.in_edges(n, data=True, keys=True) if is_multigraph else G.in_edges(n, data=True)
            for edge in in_iter:
                source = edge[0]
                data = edge[-1]
                if source not in cluster_nodes:
                    edges_to_add.append((source, new_id, data))

    # Actualización del Grafo
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
    
    if new_id in G:
        G.nodes[new_id]['x'] = avg_x
        G.nodes[new_id]['y'] = avg_y
    else:
        G.add_node(new_id, x=avg_x, y=avg_y)

    for u, v, data in edges_to_add:
        G.add_edge(u, v, **data)
            
    return G

def aplicar_snap_vertice_a_segmento(G, node_id, edge_u, edge_v, proj_x, proj_y, edge_key=0):
    """
    Fase 2: Inserta un nodo en un segmento y fusiona el nodo original en él.
    Rompe la arista (u,v) -> (u, new) y (new, v).
    """
    if node_id not in G or edge_u not in G or edge_v not in G:
        return G

    # 1. Crear el nuevo nodo de intersección
    # Usamos un ID determinista o nuevo
    new_node_id = f"snap_{node_id}_{edge_u}_{edge_v}"
    
    # Si ya existe (raro), iteramos
    count = 0
    while new_node_id in G:
        new_node_id = f"{new_node_id}_{count}"
        count += 1

    G.add_node(new_node_id, x=proj_x, y=proj_y, geometry=Point(proj_x, proj_y))

    # 2. Obtener datos de la arista original y eliminarla
    # En MultiGraph necesitamos la key
    if G.is_multigraph():
        try:
            data = G.get_edge_data(edge_u, edge_v, key=edge_key)
            G.remove_edge(edge_u, edge_v, key=edge_key)
        except KeyError:
            return G # La arista ya no existe (modificada por otro snap)
    else:
        data = G.get_edge_data(edge_u, edge_v)
        G.remove_edge(edge_u, edge_v)

    # 3. Crear las dos nuevas aristas (Split)
    # Heredamos atributos, pero la geometría debería cortarse. 
    # Por simplicidad en demo, omitimos recálculo exacto de 'geometry' LINESTRING.
    G.add_edge(edge_u, new_node_id, **data)
    G.add_edge(new_node_id, edge_v, **data)

    # 4. Mover conexiones del nodo_id al new_node_id (Merge)
    # Esencialmente, colapsamos node_id -> new_node_id
    # Reutilizamos la lógica de colapso de cluster para un set de 2 elementos
    # Pero aquí es más específico: new_node_id es el destino fijo.
    
    is_multigraph = G.is_multigraph()
    
    # Redirigir salientes de node_id
    out_iter = list(G.out_edges(node_id, data=True, keys=True)) if is_multigraph else list(G.out_edges(node_id, data=True))
    for edge in out_iter:
        target = edge[1]
        e_data = edge[-1]
        if target != new_node_id and target != node_id:
            G.add_edge(new_node_id, target, **e_data)
            
    # Redirigir entrantes a node_id
    in_iter = list(G.in_edges(node_id, data=True, keys=True)) if is_multigraph else list(G.in_edges(node_id, data=True))
    for edge in in_iter:
        source = edge[0]
        e_data = edge[-1]
        if source != new_node_id and source != node_id:
            G.add_edge(source, new_node_id, **e_data)

    # 5. Eliminar el nodo original
    G.remove_node(node_id)
    
    return G

# --- Algoritmos Principales ---

def ejecutar_fase_2_segmentos(G, R):
    """
    Detecta vértices cercanos a segmentos y aplica snaps.
    Usa ordenamiento en X para optimizar (Sweep-line simplificado).
    """
    # 1. Preparar lista de Segmentos y Vértices
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({'id': n, 'x': data['x'], 'y': data['y']})
    
    # Ordenar nodos por X
    nodes.sort(key=lambda k: k['x'])

    edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        # Obtenemos coords de extremos
        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
        vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
        # Bounding box X del segmento para filtrado rápido
        min_x = min(ux, vx)
        max_x = max(ux, vx)
        edges.append({
            'u': u, 'v': v, 'key': k, 
            'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy,
            'min_x': min_x, 'max_x': max_x
        })
    
    # Ordenar aristas por inicio X (min_x)
    edges.sort(key=lambda e: e['min_x'])

    snaps_to_perform = [] # (node_id, edge_u, edge_v, proj_x, proj_y, key, dist)
    nodes_snapped = set()
    edges_affected = set()

    # 2. Barrido / Comparación
    # Iteramos vértices y buscamos segmentos cercanos
    # Nota: Una optimización completa de sweep-line es compleja en Python puro.
    # Usaremos una ventana deslizante simple sobre la lista ordenada de edges.
    
    edge_idx = 0
    n_edges = len(edges)
    
    for node in nodes:
        nx_val, ny_val = node['x'], node['y']
        
        # Avanzar el índice de aristas mientras edge.max_x < node.x - 2R
        # (Es decir, la arista está totalmente a la izquierda del nodo y fuera de alcance)
        # Esto asume que edges están ordenados, pero edges.max_x no es monótono con min_x.
        # Simplificación: Iterar todas es costoso O(VE).
        # Optimización: Filtrar solo edges donde abs(edge.mid_x - node.x) < R + edge_len/2 ?
        # Volvamos a fuerza bruta optimizada con break si X se aleja demasiado, 
        # pero como edges tienen longitudes variables, solo podemos podar por min_x.
        
        best_snap = None
        min_dist = float('inf')

        # Buscar en aristas
        for e in edges:
            # Poda por derecha: Si el inicio del segmento está muy a la derecha del nodo
            if e['min_x'] > nx_val + 2*R:
                # Como edges están ordenados por min_x, los siguientes también estarán lejos
                break 
                
            # Poda por izquierda (manual check): Si el fin del segmento está muy a la izquierda
            if e['max_x'] < nx_val - 2*R:
                continue

            # Evitar auto-snap (si el nodo es endpoint de la arista)
            if node['id'] == e['u'] or node['id'] == e['v']:
                continue

            # Cálculo de distancia rigurosa
            d, px, py, t = distancia_punto_segmento(nx_val, ny_val, e['ux'], e['uy'], e['vx'], e['vy'])
            
            # Condición de radio: Distancia <= 2R (Radio nodo + Radio segmento)
            # Y asegurar que la proyección cae "dentro" del segmento (t no es 0 ni 1 extremo estricto)
            # Usamos un epsilon para no colapsar justo en el vértice (eso lo maneja la Fase 1)
            epsilon = 0.05
            if d <= 2 * R and epsilon < t < (1 - epsilon):
                if d < min_dist:
                    min_dist = d
                    best_snap = (node['id'], e['u'], e['v'], px, py, e['key'], d)

        if best_snap:
            # Conflicto: Si múltiples nodos quieren la misma arista, o un nodo a múltiples aristas.
            # Aquí tomamos el mejor snap para este nodo.
            snaps_to_perform.append(best_snap)

    # 3. Aplicar Snaps
    # Ordenamos por distancia (priorizar los más cercanos)
    snaps_to_perform.sort(key=lambda x: x[6])
    
    count = 0
    for snap in snaps_to_perform:
        nid, u, v, px, py, k, d = snap
        
        # Verificar que la arista o el nodo no hayan sido modificados ya
        if nid in nodes_snapped: continue
        # Si la arista fue tocada, es arriesgado dividirla de nuevo sin recalcular. 
        # Saltamos snaps concurrentes en la misma arista para esta iteración.
        edge_id = (min(u, v), max(u, v), k)
        if edge_id in edges_affected: continue
        
        G = aplicar_snap_vertice_a_segmento(G, nid, u, v, px, py, k)
        nodes_snapped.add(nid)
        edges_affected.add(edge_id)
        count += 1
        
    return G, count

def simplify_graph_complete(G_source, R):
    G = G_source.copy()
    
    # --- FASE 1: Clustering Vértice-Vértice ---
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    circles = []
    for nid, row in nodes_gdf.iterrows():
        circles.append({'id': nid, 'x': row['x'], 'y': row['y']})
    circles.sort(key=lambda c: c['x']) # Sort para barrido
    
    ds = DisjointSet(list(G.nodes()))
    
    # Barrido DSU
    for i in range(len(circles)):
        c1 = circles[i]
        for j in range(i - 1, -1, -1):
            c2 = circles[j]
            if (c1['x'] - c2['x']) > 2 * R: break
            
            dist = math.hypot(c1['x'] - c2['x'], c1['y'] - c2['y'])
            if dist <= 2 * R:
                ds.union(c1['id'], c2['id'])

    clusters = {}
    for node in G.nodes():
        root = ds.find(node)
        if root not in clusters: clusters[root] = []
        clusters[root].append(node)
    
    for root, nodes in clusters.items():
        if len(nodes) > 1:
            G = colapsar_vertices_cluster(nodes, G)

    # --- FASE 2: Colapso Vértice-Segmento ---
    # Se ejecuta después de limpiar los clusters de puntos
    G, snaps_count = ejecutar_fase_2_segmentos(G, R)
            
    return G, circles, snaps_count

# --- Interfaz Visual ---

def run_interactive_visualization():
    point = (-12.122, -77.028) 
    dist = 800 
    print(f"Descargando grafo {point} radio {dist}m...")
    
    try:
        G_raw = ox.graph_from_point(point, dist=dist, network_type='drive')
    except Exception:
        # Fallback simple si falla descarga
        G_raw = ox.graph_from_point(point, dist=dist, network_type='all')

    # Compatibilidad y Proyección
    if hasattr(ox, 'get_largest_component'):
        G_raw = ox.get_largest_component(G_raw, strongly=True)
    else:
        from osmnx import truncate
        G_raw = truncate.largest_component(G_raw, strongly=True)
        
    G_raw = ox.project_graph(G_raw)
    print(f"Nodos: {len(G_raw)}, Aristas: {len(G_raw.edges)}")

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Radio R (m)', 1, 80, valinit=15, valstep=1)
    
    state = {'R': 15}

    def update_graph(val):
        R = slider.val
        state['R'] = R
        ax.clear()
        ax.set_title(f"Procesando Fase 1 y 2... R={R}m", color='red')
        fig.canvas.draw()

        G_sim, circles_data, snaps = simplify_graph_complete(G_raw, R)
        
        ax.clear()
        # Aristas
        for u, v in G_sim.edges():
            try:
                x = [G_sim.nodes[u]['x'], G_sim.nodes[v]['x']]
                y = [G_sim.nodes[u]['y'], G_sim.nodes[v]['y']]
                ax.plot(x, y, color='black', linewidth=1, alpha=0.7, zorder=1)
            except KeyError: continue 
            
        # Nodos (Radio visual)
        for c in circles_data:
            circle = plt.Circle((c['x'], c['y']), R, color='blue', alpha=0.03, zorder=0)
            ax.add_patch(circle)
            
        ax.set_title(f"Original: {len(G_raw)} -> Simplificado: {len(G_sim)} (Snaps Segmento: {snaps}) | R={R}m")
        ax.set_aspect('equal')
        fig.canvas.draw_idle()

    def on_release(event):
        if slider.val != state['R']:
            update_graph(slider.val)

    fig.canvas.mpl_connect('button_release_event', on_release)
    update_graph(15)
    plt.show()

if __name__ == "__main__":
    run_interactive_visualization()