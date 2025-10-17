// Description: Fast point matching library using C++ backend with pybind11

// Include necessary headers
#include <cmath>
#include <limits>
#include <vector>

// Include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include CGAL headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel                  K;
typedef CGAL::Delaunay_triangulation_2<K>                                    DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT>                 AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT,AT,AP>                                    VD;

typedef AT::Site_2  Site_2;
typedef AT::Point_2 Point_2;

namespace py = pybind11;

// Match Bruteforce
py::tuple match_bruteforce(py::array_t<double, py::array::c_style | py::array::forcecast> A,
                           py::array_t<double, py::array::c_style | py::array::forcecast> B) {
    // Validate input shapes
    if (A.ndim() != 2 || A.shape(1) != 2)
        throw std::runtime_error("A must be Nx2 array");
    if (B.ndim() != 2 || B.shape(1) != 2)
        throw std::runtime_error("B must be Mx2 array");

    size_t n = static_cast<size_t>(A.shape(0));
    size_t m = static_cast<size_t>(B.shape(0));

    // Get raw data pointers
    const double* a_ptr = static_cast<const double*>(A.data());
    const double* b_ptr = static_cast<const double*>(B.data());

    // Result vectors
    std::vector<int> indices(n);
    std::vector<double> distances(n);

    // For each point in A, find nearest point in B
    for (size_t i = 0; i < n; i++) {
        double ax = a_ptr[2*i];
        double ay = a_ptr[2*i + 1];
        
        double min_distance = std::numeric_limits<double>::infinity();
        int best_index = -1;

        // Check all points in B
        for (size_t j = 0; j < m; j++) {
            double bx = b_ptr[2*j];
            double by = b_ptr[2*j + 1];
            
            // Calculate Euclidean distance
            double dx = ax - bx;
            double dy = ay - by;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                best_index = static_cast<int>(j);
            }
        }
        
        indices[i] = best_index;
        distances[i] = min_distance;
    }

    // Convert to numpy arrays
    py::array_t<int> result_indices = py::cast(indices);
    py::array_t<double> result_distances = py::cast(distances);

    return py::make_tuple(result_indices, result_distances);
}

// Match with CGAL
py::tuple match_cgal(py::array_t<double, py::array::c_style | py::array::forcecast> A,
                     py::array_t<double, py::array::c_style | py::array::forcecast> B) {
    // Validate input shapes
    if (A.ndim() != 2 || A.shape(1) != 2)
        throw std::runtime_error("A must be Nx2 array");
    if (B.ndim() != 2 || B.shape(1) != 2)
        throw std::runtime_error("B must be Mx2 array");

    size_t n = static_cast<size_t>(A.shape(0));
    size_t m = static_cast<size_t>(B.shape(0));

    // Get raw data pointers
    const double* a_ptr = static_cast<const double*>(A.data());
    const double* b_ptr = static_cast<const double*>(B.data());

    // Result vectors
    std::vector<int> indices(n);
    std::vector<double> distances(n);

    // Insert points from B into Delaunay triangulation and map for indices
    DT dt;
    std::map<Point_2, int> point_index_map;

    for (size_t j = 0; j < m; j++) {
        Point_2 p(b_ptr[2*j], b_ptr[2*j + 1]);

        dt.insert(p);
        point_index_map[p] = static_cast<int>(j);
    }

    // For each point in A, find nearest point in B using CGAL
    for (size_t i = 0; i < n; i++) {
        Point_2 query(a_ptr[2*i], a_ptr[2*i + 1]);
        auto nearest_vertex = dt.nearest_vertex(query);

        // Calculate distance
        double dx = a_ptr[2*i] - nearest_vertex->point().x();
        double dy = a_ptr[2*i + 1] - nearest_vertex->point().y();
        double distance = std::sqrt(dx*dx + dy*dy);

        // Get index from map
        int best_index = point_index_map[nearest_vertex->point()];

        indices[i] = best_index;
        distances[i] = distance;
    }

    // Convert to numpy arrays
    py::array_t<int> result_indices = py::cast(indices);
    py::array_t<double> result_distances = py::cast(distances);

    return py::make_tuple(result_indices, result_distances);
}

// Python module definition
PYBIND11_MODULE(matcher, m) {
    m.doc() = "Fast point matching library using C++ backend";
    
    m.def("match_bruteforce", &match_bruteforce,
          "Find nearest point in B for each point in A using brute force algorithm",
          py::arg("A"), py::arg("B"));
    
    m.def("match_cgal", &match_cgal,
          "Find nearest point in B for each point in A using CGAL Delaunay triangulation",
          py::arg("A"), py::arg("B"));
}
