#pragma once
#include "types.hpp"

SimplificationResult simplify_graph_topological_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments
);

SimplificationResult simplify_graph_geometric_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double threshold
);

SimplificationResult simplify_graph_parallel_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double distance_threshold,
    double angle_threshold_deg
);

SimplificationResult simplify_graph_minkowski_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double radius
);

SimplificationResult minkowski_guided_simplify(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes_y,
    py::array_t<long> seg_start,
    py::array_t<long> seg_end,
    double radius
);

SimplificationResult simplify_graph_acj_master_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double angulo_maximo_desviacion,
    double factor_ajuste,
    double factor_epsilon,
    bool with_index
);
