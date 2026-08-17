"""Thesis output generation: matplotlib plots and CSV exports organised by city."""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Colour palette keyed by algorithm display name ────────────────────────────
_PALETTE = {
    "Raw OSM": "#6C757D",  # Gris suave para la referencia
    "OSMnx": "#D90429",  # Rojo
    "NeatNet": "#F4A261",  # Naranja
    "GeoJAC": "#2A9D8F",  # Verde
}
_FALLBACK = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]


def _city_slug(city_name: str) -> str:
    return city_name.replace(",", "").replace(" ", "_").lower()


def _color(name: str, idx: int) -> str:
    return _PALETTE.get(name, _FALLBACK[idx % len(_FALLBACK)])


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
            bars = ax.bar(labels, vals, color=colors, edgecolor="black")
            ax.set_title(title)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(
                labels, rotation=15, ha="right", fontsize=9, fontweight="bold"
            )
            offset = max(vals) * 0.01 if max(vals) > 0 else 0.5
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_metrics_basic.png", show)

        # ── 2. Sinuosity vs Coords (Trade-off Scatter Plot) ──────────────────
        fig, ax = plt.subplots(figsize=(10, 6))

        raw_sinuosity = results.get("Raw OSM", {}).get("avg_sinuosity", 1.0)

        for i, lbl in enumerate(labels):
            x_val = results[lbl].get("coords", 0)
            y_val = results[lbl].get("avg_sinuosity", 1.0)
            c_val = _color(lbl, i)

            # Dibujar la referencia Raw OSM como una estrella y con línea guía
            if lbl == "Raw OSM":
                ax.scatter(
                    x_val,
                    y_val,
                    color=c_val,
                    s=300,
                    marker="*",
                    edgecolor="black",
                    label=f"{lbl} (Ref)",
                    zorder=5,
                )
                ax.axhline(y=y_val, color=c_val, linestyle="--", alpha=0.7, zorder=1)
            else:
                ax.scatter(
                    x_val,
                    y_val,
                    color=c_val,
                    s=150,
                    marker="o",
                    edgecolor="black",
                    label=lbl,
                    zorder=5,
                )

            # Añadir etiquetas de texto a cada punto
            ax.text(
                x_val,
                y_val,
                f"  {lbl}\n  (Sin: {y_val:.3f})",
                fontsize=10,
                va="center",
                ha="left",
                fontweight="bold",
                zorder=6,
            )

        ax.set_title(
            f"Trade-off: Simplificación vs Geometría Original\n{self.city_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Total de Coordenadas (Menos coordenadas = Mayor compresión)",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_ylabel("Sinuosidad Promedio", fontweight="bold", fontsize=11)

        # Ajustar el eje Y para que se note claramente la diferencia de la línea 1.0 matemática
        y_vals = [results[lbl].get("avg_sinuosity", 1.0) for lbl in labels]
        ax.set_ylim(bottom=0.99, top=max(y_vals) * 1.02)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        plt.tight_layout()
        self._savefig(fig, f"{self._slug}_sinuosity_scatter.png", show)

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

        # Colores personalizados para los errores pero manteniendo el espíritu de la tesis
        ax.bar(
            x_pos - w / 2,
            med_vals,
            w,
            label="Mediana del Error (Típico)",
            color="#343A40",
            edgecolor="black",
            alpha=0.8,
        )
        ax.bar(
            x_pos + w / 2,
            p95_vals,
            w,
            label="Percentil 95 (Peor Caso)",
            color="#ADB5BD",
            edgecolor="black",
            alpha=0.8,
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
