/**
 * ACJ Core - CGAL-based spatial indexing for point-to-graph assignment
 * 
 * This module provides high-performance spatial queries using CGAL
 * (Computational Geometry Algorithms Library). It wraps CGAL's Delaunay
 * triangulation for efficient nearest-neighbor queries.
 * 
 * Based on work by Alejandro.
 * 
 * Functions:
 *   - match_cgal: Find nearest point in target set for each query point
 *   - assign_to_segments: (PENDING) Find nearest line segment for each point
 */

#include <cmath>
#include <limits>
#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>

// CGAL type definitions
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> DT;
typedef K::Point_2 Point_2;

namespace py = pybind11;


/**
 * Find nearest neighbor in target set for each query point using CGAL.
 * 
 * This function builds a Delaunay triangulation from the target points
 * and performs efficient nearest-neighbor queries for each query point.
 * 
 * Complexity: O(M log M) for construction + O(N log M) for N queries
 * where N = number of query points, M = number of target points.
 * 
 * @param query_points: Nx2 array of query point coordinates
 * @param target_points: Mx2 array of target point coordinates
 * @return: Tuple of (indices, distances)
 *          - indices: Array of length N with target indices
 *          - distances: Array of length N with Euclidean distances
 */
py::tuple match_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_points
) {
    // Validate input array shapes
    if (query_points.ndim() != 2 || query_points.shape(1) != 2) {
        throw std::runtime_error("query_points must be Nx2 array");
    }
    if (target_points.ndim() != 2 || target_points.shape(1) != 2) {
        throw std::runtime_error("target_points must be Mx2 array");
    }

    size_t n_query = static_cast<size_t>(query_points.shape(0));
    size_t n_target = static_cast<size_t>(target_points.shape(0));

    // Get raw data pointers for efficient access
    const double* query_ptr = static_cast<const double*>(query_points.data());
    const double* target_ptr = static_cast<const double*>(target_points.data());

    // Result containers
    std::vector<int> indices(n_query);
    std::vector<double> distances(n_query);

    // Build Delaunay triangulation from target points
    DT dt;
    std::map<Point_2, int> point_index_map;

    // Insert all target points into the triangulation
    for (size_t j = 0; j < n_target; j++) {
        Point_2 p(target_ptr[2*j], target_ptr[2*j + 1]);
        dt.insert(p);
        point_index_map[p] = static_cast<int>(j);
    }

    // Query nearest neighbor for each query point
    for (size_t i = 0; i < n_query; i++) {
        Point_2 query(query_ptr[2*i], query_ptr[2*i + 1]);
        
        // Find nearest vertex in triangulation (O(log M) operation)
        auto nearest_vertex = dt.nearest_vertex(query);
        
        // Calculate Euclidean distance
        double dx = query_ptr[2*i] - nearest_vertex->point().x();
        double dy = query_ptr[2*i + 1] - nearest_vertex->point().y();
        double distance = std::sqrt(dx*dx + dy*dy);
        
        // Retrieve original index from map
        int target_idx = point_index_map[nearest_vertex->point()];
        
        indices[i] = target_idx;
        distances[i] = distance;
    }

    // Convert results to numpy arrays
    py::array_t<int> result_indices = py::cast(indices);
    py::array_t<double> result_distances = py::cast(distances);

    return py::make_tuple(result_indices, result_distances);
}


/**
 * PENDING: Assign points to nearest line segments using AABB tree.
 * 
 * This function will use CGAL's AABB tree to efficiently find the closest
 * line segment for each query point and compute the projection point.
 * 
 * Implementation requires:
 *   - CGAL AABB tree construction from segments
 *   - Point-to-segment distance queries
 *   - Projection point computation
 * 
 * @param query_points: Nx2 array of query point coordinates
 * @param segments: Mx4 array of segment endpoints (x1, y1, x2, y2)
 * @return: Tuple of (indices, distances, projections)
 * 
 * Status: NOT IMPLEMENTED
 */
py::tuple assign_to_segments(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments
) {
    throw std::runtime_error(
        "assign_to_segments() is not yet implemented. "
        "This requires CGAL AABB tree support for line segments."
    );
    
    // PENDING IMPLEMENTATION:
    // 1. Build AABB tree from segments using CGAL::AABB_tree
    // 2. For each query point, find closest segment
    // 3. Compute projection point on segment
    // 4. Return (segment_indices, distances, projection_points)
}


/**
 * Python module definition for acj_core.
 * 
 * This module is imported by the Python MapIndex class to perform
 * high-performance spatial queries.
 */
PYBIND11_MODULE(acj_core, m) {
    m.doc() = "ACJ Core - CGAL-based spatial indexing for geospatial analysis";
    
    m.def("match_cgal", &match_cgal,
          "Find nearest target point for each query point using CGAL Delaunay triangulation.\n\n"
          "Args:\n"
          "    query_points: Nx2 numpy array of query coordinates\n"
          "    target_points: Mx2 numpy array of target coordinates\n\n"
          "Returns:\n"
          "    Tuple of (indices, distances) where:\n"
          "    - indices[i] is the index in target_points of the nearest neighbor\n"
          "    - distances[i] is the Euclidean distance to that neighbor",
          py::arg("query_points"), 
          py::arg("target_points"));
    
    m.def("assign_to_segments", &assign_to_segments,
          "Assign points to nearest line segments (PENDING IMPLEMENTATION).\n\n"
          "Args:\n"
          "    query_points: Nx2 numpy array of query coordinates\n"
          "    segments: Mx4 numpy array of segment endpoints (x1, y1, x2, y2)\n\n"
          "Returns:\n"
          "    Tuple of (indices, distances, projections)",
          py::arg("query_points"),
          py::arg("segments"));
}
