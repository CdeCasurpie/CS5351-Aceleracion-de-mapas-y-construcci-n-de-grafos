"""
generate_thesis_figures.py
==========================
Genera 3 figuras haciendo ZOOM en una zona crítica.
Comparación lado a lado (Original vs Simplificado).

Uso: python3 scripts/generate_thesis_figures.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox

# Si falla la importación de geojac/neatnet en tu entorno local, ajusta estos imports
from geojac import UrbanNetwork, ACJTopologicalEvaluator, CompressionRatioMetric

CITY    = "Barranco, Lima, Peru"
OUT_DIR = "./outputs/thesis_figures"
DPI     = 300  # Alta resolución para la tesis

# --- NUEVA PALETA Y ESTILOS ---
C_ORIGINAL   = "#000000"  # Negro puro para el grafo original
C_ERROR      = "#D90429"  # Rojo fuerte para OSMnx/NeatNet (Fallo)
C_SUCCESS    = "#2A9D8F"  # Verde esmeralda para GeoJAC (Éxito)
BG           = "#F8F9FA"  # Fondo ligeramente off-white

matplotlib.rcParams.update({
    "font.family": "sans-serif", 
    "axes.facecolor": BG, "figure.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
})
os.makedirs(OUT_DIR, exist_ok=True)

# ── FUNCIONES DE DIBUJO ───────────────────────────────────────────────────────

def plot_original_graph(ax, G, color):
    """Dibuja el grafo original en negro con nodos."""
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        xs, ys = (geom.xy if geom else
                  ([G.nodes[u]["x"], G.nodes[v]["x"]],
                   [G.nodes[u]["y"], G.nodes[v]["y"]]))
        ax.plot(xs, ys, color=color, lw=2.0, zorder=1)
        
    # Nodos originales
    nx_xs = [d["x"] for _, d in G.nodes(data=True)]
    nx_ys = [d["y"] for _, d in G.nodes(data=True)]
    ax.scatter(nx_xs, nx_ys, color=color, s=15, zorder=2, edgecolor='white', linewidth=0.5)

def plot_nx_simplified(ax, G_simp, color):
    """Dibuja el grafo simplificado como una línea (OSMnx)."""
    for u, v, d in G_simp.edges(data=True):
        geom = d.get("geometry")
        xs, ys = (geom.xy if geom else
                  ([G_simp.nodes[u]["x"], G_simp.nodes[v]["x"]],
                   [G_simp.nodes[u]["y"], G_simp.nodes[v]["y"]]))
        ax.plot(xs, ys, color=color, lw=2.5, zorder=4)
        
    nx_xs = [d["x"] for _, d in G_simp.nodes(data=True)]
    nx_ys = [d["y"] for _, d in G_simp.nodes(data=True)]
    ax.scatter(nx_xs, nx_ys, color=color, s=20, zorder=5, edgecolor='white')

def plot_gdf_simplified(ax, edges_gdf, color):
    """Dibuja el GeoDataFrame de NeatNet e infiere los nodos en los extremos."""
    nodes_x, nodes_y = [], []
    
    for geom in edges_gdf.geometry:
        if geom is None: continue
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            ax.plot(xs, ys, color=color, lw=2.5, zorder=4)
            # Extraer nodos extremos
            nodes_x.extend([xs[0], xs[-1]])
            nodes_y.extend([ys[0], ys[-1]])
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=color, lw=2.5, zorder=4)
                # Extraer nodos extremos
                nodes_x.extend([xs[0], xs[-1]])
                nodes_y.extend([ys[0], ys[-1]])
                
    # Dibujar los nodos para NeatNet
    ax.scatter(nodes_x, nodes_y, color=color, s=20, zorder=5, edgecolor='white')

def plot_urban_simplified(ax, net, color):
    """Dibuja el grafo de GeoJAC y sus nodos."""
    xy = net.nodes_df.set_index("node_id")[["x", "y"]]
    for _, row in net.edges_df.iterrows():
        u, v = int(row["node_start"]), int(row["node_end"])
        if u in xy.index and v in xy.index:
            ax.plot([xy.loc[u,"x"], xy.loc[v,"x"]],
                    [xy.loc[u,"y"], xy.loc[v,"y"]],
                    color=color, lw=2.5, zorder=4)
            
    ax.scatter(net.nodes_df["x"], net.nodes_df["y"], color=color, s=20, zorder=5, edgecolor='white')

def apply_zoom(ax, G_raw):
    """Enfoca la cámara en una zona densa y amplia de Barranco."""
    xs = [d["x"] for _, d in G_raw.nodes(data=True)]
    ys = [d["y"] for _, d in G_raw.nodes(data=True)]
    
    # Desplazamos un poco el centro para agarrar una zona de calles más irregulares
    cx, cy = np.percentile(xs, 55), np.percentile(ys, 45)
    
    # Radio de 450 metros (área visible de 900x900m)
    zoom_radius = 450 
    ax.set_xlim(cx - zoom_radius, cx + zoom_radius)
    ax.set_ylim(cy - zoom_radius, cy + zoom_radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def make_figure(fname, title, G_raw, draw_simp_func, color, label_alg):
    # Figura más ancha para colocar los mapas lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.02, right=0.98, wspace=0.05)
    
    fig.suptitle(title, fontsize=20, fontweight="bold")
    
    # Lado Izquierdo: Original
    ax1.set_title("Grafo Original", fontsize=15, pad=10)
    plot_original_graph(ax1, G_raw, C_ORIGINAL)
    apply_zoom(ax1, G_raw)
    
    # Lado Derecho: Simplificado
    ax2.set_title(f"Simplificado ({label_alg})", fontsize=15, pad=10)
    draw_simp_func(ax2, color)
    apply_zoom(ax2, G_raw)
    
    # Leyenda global ubicada en el centro inferior de toda la figura
    legend_elements = [
        Line2D([0], [0], color=C_ORIGINAL, lw=2.0, marker='o', markersize=5, label='Original'),
        Line2D([0], [0], color=color, lw=2.5, marker='o', markersize=6, label=label_alg)
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, frameon=False, fontsize=14)
    
    for fmt in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{fname}.{fmt}", format=fmt, dpi=DPI)
    plt.close()
    print(f"✅ Generado: {fname}.png")

def make_intro_figure(fname, G_raw, draw_simp_func, color):
    """Genera una figura limpia de una sola columna (ideal para el slide de introducción)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    
    # Dibujamos solo la red simplificada
    draw_simp_func(ax, color)
    apply_zoom(ax, G_raw)
    
    # Leyenda sutil
    legend_elements = [
        Line2D([0], [0], color=color, lw=3.0, marker='o', markersize=8, label="Red Simplificada")
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor=BG, edgecolor="none", fontsize=14)
    
    for fmt in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{fname}.{fmt}", format=fmt, dpi=DPI, transparent=True)
    plt.close()
    print(f"✅ Generado: {fname}.png (Slide Introductorio)")


# ── DESCARGA Y PROCESAMIENTO ──────────────────────────────────────────────────
print(f"\n[1] Descargando {CITY} …")
# Proyectamos a UTM para trabajar en metros reales
G_geo = ox.graph_from_place(CITY, network_type="drive", simplify=False)
G_raw = ox.project_graph(G_geo)

print("[2] Ejecutando OSMnx …")
G_osmnx = ox.simplify_graph(G_raw.copy())

print("[3] Ejecutando NeatNet …")
neatnet_gdf = None
try:
    import neatnet
    _, edges_raw = ox.graph_to_gdfs(G_raw)
    edges_clean  = edges_raw[["geometry"]].reset_index(drop=True)
    neatnet_gdf  = neatnet.neatify(edges_clean)
except Exception as e:
    print(f"    [WARN] NeatNet no disponible: {e}")

print("[4] Ejecutando GeoJAC …")
network   = UrbanNetwork.from_networkx(G_raw)
evaluator = ACJTopologicalEvaluator(network, [CompressionRatioMetric()])
evaluator.evaluate()
geo_net = evaluator.simplified_network

# ── GENERACIÓN DE FIGURAS ─────────────────────────────────────────────────────

# FIG 1: OSMnx
make_figure(
    fname="fig_osmnx_zoom",
    title="OSMnx",
    G_raw=G_raw,
    draw_simp_func=lambda ax, c: plot_nx_simplified(ax, G_osmnx, c),
    color=C_ERROR,
    label_alg="OSMnx"
)

# FIG 2: NeatNet
if neatnet_gdf is not None:
    make_figure(
        fname="fig_neatnet_zoom",
        title="NeatNet",
        G_raw=G_raw,
        draw_simp_func=lambda ax, c: plot_gdf_simplified(ax, neatnet_gdf, c),
        color=C_ERROR,
        label_alg="NeatNet"
    )

# FIG 3: GeoJAC
make_figure(
    fname="fig_geojac_zoom",
    title="GEOJAC",
    G_raw=G_raw,
    draw_simp_func=lambda ax, c: plot_urban_simplified(ax, geo_net, c),
    color=C_SUCCESS,
    label_alg="GEOJAC"
)

# FIG 0: INTRODUCCIÓN (Solo GeoJAC, un solo cuadro)
make_intro_figure(
    fname="fig_geojac_intro",
    G_raw=G_raw,
    draw_simp_func=lambda ax, c: plot_urban_simplified(ax, geo_net, c),
    color=C_SUCCESS
)

print("\n🎓 ¡Figuras listas! Revisa la carpeta outputs/thesis_figures.")
