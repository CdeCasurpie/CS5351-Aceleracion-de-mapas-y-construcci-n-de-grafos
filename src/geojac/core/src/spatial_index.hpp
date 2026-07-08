#pragma once
#include "types.hpp"

py::tuple match_point(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_points
);

py::list find_clusters_cgal(
     py::array_t<double, py::array::c_style | py::array::forcecast> points,
     double threshold
);

py::tuple match_segment(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments
);
