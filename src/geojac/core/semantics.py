import pandas as pd
from geojac.core.network import UrbanNetwork
from geojac.data.io import SimplificationResult

def resolve_semantics(original: UrbanNetwork, result: SimplificationResult) -> UrbanNetwork:
    resolved = UrbanNetwork()
    
    # Transfer topology
    resolved.nodes_df = result.graph.nodes.copy()
    resolved.edges_df = result.graph.segments.copy()
    
    # Transfer lineage
    resolved.lineage_nodes = result.node_lineage
    resolved.lineage_edges = result.edge_lineage
    
    def merge_metadata(lineage, original_metadata):
        new_metadata = {}
        for new_id, old_ids in lineage.items():
            valid_metas = [original_metadata[old_id] for old_id in old_ids if old_id in original_metadata]
            if not valid_metas:
                continue
            
            merged = {}
            keys = set()
            for m in valid_metas:
                keys.update(m.keys())
                
            for k in keys:
                vals = [m[k] for m in valid_metas if k in m]
                if not vals:
                    continue
                
                if isinstance(vals[0], bool):
                    merged[k] = any(vals)
                elif isinstance(vals[0], (int, float)):
                    merged[k] = sum(vals) / len(vals)
                elif isinstance(vals[0], str):
                    # Use dict.fromkeys to keep unique elements in order
                    unique_vals = list(dict.fromkeys(vals))
                    merged[k] = " | ".join(unique_vals)
                else:
                    unique_vals = list(dict.fromkeys((str(v) for v in vals)))
                    merged[k] = " | ".join(unique_vals)
            
            new_metadata[new_id] = merged
        return new_metadata

    resolved.node_metadata = merge_metadata(result.node_lineage, original.node_metadata)
    resolved.edge_metadata = merge_metadata(result.edge_lineage, original.edge_metadata)
    
    return resolved
