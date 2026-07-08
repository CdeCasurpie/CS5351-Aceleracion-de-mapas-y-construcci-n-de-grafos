import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from geojac.core.network import UrbanNetwork
from geojac.evaluation.metrics import CompressionRatioMetric, SemanticSpeedDistortionMetric
from geojac.evaluation.evaluators import ACJTopologicalEvaluator

def test_evaluator_and_metrics():
    original = UrbanNetwork()
    
    original.nodes_df = pd.DataFrame({
        'node_id': [0, 1, 2, 3],
        'x': [0.0, 10.0, 20.0, 30.0],
        'y': [0.0, 0.0, 0.0, 0.0]
    })
    
    original.edges_df = pd.DataFrame({
        'segment_id': [100, 101, 102],
        'node_start': [0, 1, 2],
        'node_end': [1, 2, 3]
    })
    
    original.edge_metadata = {
        100: {'maxspeed': 50},
        101: {'maxspeed': 60},
        102: {'maxspeed': 70}
    }
    
    metrics = [CompressionRatioMetric(), SemanticSpeedDistortionMetric()]
    evaluator = ACJTopologicalEvaluator(original, metrics)
    
    results = evaluator.evaluate()
    
    assert isinstance(results, dict)
    assert 'CompressionRatio' in results
    assert 'SemanticSpeedDistortion' in results
    assert isinstance(results['CompressionRatio'], float)
    assert isinstance(results['SemanticSpeedDistortion'], float)
    
    # Original has 4 nodes. Topological simplification removes nodes 1 and 2 (degree 2)
    # So 2 nodes remain. 1.0 - (2 / 4) = 0.5
    assert results['CompressionRatio'] == 0.5
