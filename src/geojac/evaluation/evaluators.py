import pandas as pd
import numpy as np
from geojac.evaluation.base import BaseEvaluator
from geojac.data.io import SimplificationResult, GraphData


class ACJTopologicalEvaluator(BaseEvaluator):
    def simplify(self) -> SimplificationResult:
        try:
            import sys
            import os

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            build_root = os.path.join(project_root, "build")
            build_core = os.path.join(build_root, "src", "geojac", "core")

            for path in [build_root, build_core]:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)

            import acj_core
        except ImportError as e:
            raise ImportError("CGAL core module not available.") from e

        nodes_df = self.original_network.nodes_df
        segments_df = self.original_network.edges_df

        nodes_array = np.ascontiguousarray(
            nodes_df[["node_id", "x", "y"]].values, dtype=np.float64
        )
        segments_array = np.ascontiguousarray(
            segments_df[["segment_id", "node_start", "node_end"]].values,
            dtype=np.float64,
        )

        cgal_result = acj_core._experimental.simplify_graph_topological_cgal(
            nodes_array, segments_array
        )

        nodes_list, segments_list = cgal_result.graph

        if not nodes_list:
            new_nodes_df = pd.DataFrame(columns=["node_id", "x", "y"])
        else:
            new_nodes_df = pd.DataFrame(nodes_list, columns=["node_id", "x", "y"])

        if not segments_list:
            new_segments_df = pd.DataFrame(
                columns=["segment_id", "node_start", "node_end", "x1", "y1", "x2", "y2"]
            )
        else:
            new_segments_df = pd.DataFrame(
                segments_list,
                columns=[
                    "segment_id",
                    "node_start",
                    "node_end",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )

        g = GraphData(new_nodes_df, new_segments_df)

        # In C++, node_lineage is a dict-like from pybind, we convert it to dict just in case
        node_lineage = (
            dict(cgal_result.node_lineage)
            if hasattr(cgal_result, "node_lineage")
            else {}
        )
        edge_lineage = (
            dict(cgal_result.edge_lineage)
            if hasattr(cgal_result, "edge_lineage")
            else {}
        )

        return SimplificationResult(
            g, node_lineage=node_lineage, edge_lineage=edge_lineage
        )
