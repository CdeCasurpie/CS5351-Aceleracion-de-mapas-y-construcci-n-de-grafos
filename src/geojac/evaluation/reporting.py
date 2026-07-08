"""Thesis output generation: matplotlib plots and CSV exports organised by city."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from geojac.core.network import UrbanNetwork

# ── Colour palette keyed by algorithm display name ────────────────────────────
_PALETTE = {
    "Raw OSM": "black",
    "OSMnx Standard": "orange",
    "NeatNet Morfológico": "green",
    "ACJ Topology+DP": "magenta",
}
_FALLBACK = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]


def _city_slug(city_name: str) -> str:
    return city_name.replace(",", "").replace(" ", "_").lower()


def _color(name: str, idx: int) -> str:
    return _PALETTE.get(name, _FALLBACK[idx % len(_FALLBACK)])


def _ax_clean(ax):
    ax.set_facecolor("white")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_network(
    ax, network: UrbanNetwork, color: str, lw: float = 0.5, ms: float = 2.0
) -> None:
    """Render UrbanNetwork onto ax using edges_df + nodes_df coordinates."""
    coords = network.nodes_df.set_index("node_id")[["x", "y"]]
    for _, row in network.edges_df.iterrows():
        u, v = int(row["node_start"]), int(row["node_end"])
        if u in coords.index and v in coords.index:
            ax.plot(
                [coords.loc[u, "x"], coords.loc[v, "x"]],
                [coords.loc[u, "y"], coords.loc[v, "y"]],
                color=color,
                linewidth=lw,
                alpha=0.7,
            )
    ax.scatter(coords["x"], coords["y"], c=color, s=ms, zorder=3)


# ── ThesisReportGenerator ─────────────────────────────────────────────────────


class ThesisReportGenerator:
    """
    Saves benchmark outputs for a single city under:
      <output_base>/<city_slug>/plots/
      <output_base>/<city_slug>/metrics/
    All plot methods accept show=False (default) for headless/CI use.
    """

    def __init__(self, city_name: str, output_base: str = "outputs"):
        self.city_name = city_name
        self._slug = _city_slug(city_name)
        self.plots_dir = Path(output_base) / self._slug / "plots"
        self.metrics_dir = Path(output_base) / self._slug / "metrics"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _plt():
        try:
            import matplotlib.pyplot as plt

            return plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot generation. "
                "Install it with: pip install matplotlib"
            ) from e

    def _savefig(self, fig, filename: str, show: bool) -> Path:
        plt = self._plt()
        p = self.plots_dir / filename
        fig.savefig(p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"  saved → {p}")
        return p

    # ── Public API ────────────────────────────────────────────────────────────

    def save_graph_comparison(
        self,
        networks: dict,  # {display_name: UrbanNetwork}
        show: bool = False,
    ) -> Path:
        """2×2 (or 1×N) side-by-side graph plot for all competitors."""
        plt = self._plt()
        names = list(networks.keys())
        n = len(names)
        cols = min(n, 4)
        rows = math.ceil(n / cols) if n > 4 else (2 if n == 4 else 1)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 7, rows * 7),
            facecolor="white",
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).flatten()

        _TITLES = {
            "Raw OSM": ("1. Nodos Raw de OSM", "black"),
            "OSMnx Standard": ("2. Simplificación OSMnx", "orange"),
            "NeatNet Morfológico": ("3. Simplificación NeatNet", "green"),
            "ACJ Topology+DP": ("4. Geometría ACJ (Voronoi+DP)", "magenta"),
        }

        for idx, (name, net) in enumerate(networks.items()):
            ax = axes[idx]
            _ax_clean(ax)
            color = _color(name, idx)
            _draw_network(ax, net, color)
            title_text, title_color = _TITLES.get(name, (name, color))
            ax.set_title(title_text, color=title_color, fontsize=14, fontweight="bold")

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(self.city_name, fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        return self._savefig(fig, f"{self._slug}_graphs_comparison.png", show)

    def save_metrics_plots(
        self,
        dual_results: dict,  # {"sin_blindaje": {...}, "con_blindaje": {...}}
        show: bool = False,
    ) -> None:
        """Save 5 metric plots using the con_blindaje scenario (calibrated weights)."""
        plt = self._plt()
        results = dual_results["con_blindaje"]
        labels = list(results.keys())
        colors = [_color(lbl, i) for i, lbl in enumerate(labels)]

        # ── 1. Basic metrics ─────────────────────────────────────────────────
        basic_keys = ["nodes", "edges", "coords", "avg_degree"]
        basic_titles = ["Nodos", "Aristas", "Coordenadas Totales", "Grado Promedio"]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"Métricas Básicas — {self.city_name}", fontsize=14, fontweight="bold"
        )
        for ax, key, title in zip(axes, basic_keys, basic_titles):
            vals = [results[lbl].get(key, 0) for lbl in labels]
            bars = ax.bar(labels, vals, color=colors)
            ax.set_title(title)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
            offset = max(vals) * 0.01 if max(vals) > 0 else 0.5
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_metrics_basic.png", show)

        # ── 2. Sinuosity vs Coords ───────────────────────────────────────────
        comp_labels = [lbl for lbl in labels if lbl != "Raw OSM"]
        sin_raw = results.get("Raw OSM", {}).get("avg_sinuosity", 1.0)
        comp_coords = [results[k].get("coords", 0) for k in comp_labels]
        comp_sin = [results[k].get("avg_sinuosity", 1.0) for k in comp_labels]
        x = np.arange(len(comp_labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(
            x - width / 2,
            comp_coords,
            width,
            label="Total Coordenadas",
            color="#1f77b4",
        )
        ax1.set_ylabel(
            "Total de Coordenadas (Vértices)", color="#1f77b4", fontweight="bold"
        )
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.axhline(
            y=sin_raw, color="black", linestyle="--", label="Sinuosidad Original (Raw)"
        )
        ax2.bar(
            x + width / 2, comp_sin, width, label="Sinuosidad Promedio", color="#ff7f0e"
        )
        ax2.set_ylabel("Sinuosidad Promedio", color="#ff7f0e", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        if comp_sin:
            ax2.set_ylim(bottom=1.0, top=max(comp_sin + [sin_raw]) * 1.02)
        ax1.set_title(
            f"Benchmark: Coordenadas vs Sinuosidad\n{self.city_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(comp_labels, fontweight="bold")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labs1 + labs2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
        )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_sinuosity_coords.png", show)

        # ── 3. Keypoint Displacement (TKD) ───────────────────────────────────
        tkd_vals = [results[lbl].get("keypoint_displacement_m", 0.0) for lbl in labels]
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, tkd_vals, color=colors, edgecolor="black")
        ax.set_title(
            f"Desplazamiento de Intersecciones (TKD)\nMenor es mejor — {self.city_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Error Geográfico Promedio (Metros)", fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
        ax.axhline(
            y=2.0,
            color="red",
            linestyle=":",
            linewidth=2,
            label="Límite aceptable GPS (~2m)",
        )
        ax.legend()
        offset = max(tkd_vals) * 0.01 if max(tkd_vals) > 0 else 0.1
        for bar, val in zip(bars, tkd_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.2f} m",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_keypoint_displacement.png", show)

        # ── 4. Reachability Preservation ─────────────────────────────────────
        reach_vals = [
            results[lbl].get("reachability_preservation_%", 100.0) for lbl in labels
        ]
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, reach_vals, color=colors, edgecolor="black")
        ax.set_title(
            f"Preservación de Alcanzabilidad (Reachability)\nMayor es mejor — {self.city_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Rutas Preservadas (%)", fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
        ax.set_ylim(0, 115)
        ax.axhline(
            y=100.0,
            color="green",
            linestyle="--",
            linewidth=2,
            label="100% Preservación",
        )
        ax.legend()
        for bar, val in zip(bars, reach_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_reachability.png", show)

        # ── 5. Path Error (median + P95) ─────────────────────────────────────
        med_vals = [results[lbl].get("path_error_abs_median", 0.0) for lbl in labels]
        p95_vals = [results[lbl].get("path_error_abs_p95", 0.0) for lbl in labels]
        x_pos, w = np.arange(len(labels)), 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x_pos - w / 2,
            med_vals,
            w,
            label="Mediana del Error (Típico)",
            color="#1f77b4",
            edgecolor="black",
        )
        ax.bar(
            x_pos + w / 2,
            p95_vals,
            w,
            label="Percentil 95 (Peor Caso)",
            color="#d62728",
            edgecolor="black",
        )
        ax.set_title(
            f"Error Absoluto en Caminos Mínimos\n(Pesos Geométricos Recalibrados) — {self.city_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Error Absoluto (Metros)", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        off_m = max(med_vals) * 0.02 if max(med_vals) > 0 else 0.5
        off_p = max(p95_vals) * 0.02 if max(p95_vals) > 0 else 0.5
        for i, lbl in enumerate(labels):
            ax.text(
                x_pos[i] - w / 2,
                med_vals[i] + off_m,
                f"{med_vals[i]:.2f}m",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                x_pos[i] + w / 2,
                p95_vals[i] + off_p,
                f"{p95_vals[i]:.2f}m",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_path_robust_errors.png", show)

    def save_metrics_csv(self, dual_results: dict) -> Path:
        """Flatten dual_results to CSV at <metrics_dir>/<slug>_metrics_benchmark.csv."""
        rows = []
        for escenario, scenario_data in dual_results.items():
            for model, m in scenario_data.items():
                rows.append(
                    {
                        "city": self.city_name,
                        "escenario": escenario,
                        "model": model,
                        **m,
                    }
                )
        df = pd.DataFrame(rows)
        p = self.metrics_dir / f"{self._slug}_metrics_benchmark.csv"
        df.to_csv(p, index=False)
        print(f"  saved → {p}")
        return p
