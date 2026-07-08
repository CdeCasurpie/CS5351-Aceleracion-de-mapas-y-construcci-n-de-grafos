#include "spatial_index.hpp"

py::tuple match_point(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_points
) {
    if (query_points.ndim() != 2 || query_points.shape(1) != 2) {
        throw std::runtime_error("query_points must be Nx2 array");
    }
    if (target_points.ndim() != 2 || target_points.shape(1) != 2) {
        throw std::runtime_error("target_points must be Mx2 array");
    }
    size_t n_query = static_cast<size_t>(query_points.shape(0));
    size_t n_target = static_cast<size_t>(target_points.shape(0));
    const double* query_ptr = static_cast<const double*>(query_points.data());
    const double* target_ptr = static_cast<const double*>(target_points.data());

    std::vector<int> indices(n_query);
    std::vector<double> distances(n_query);
    DT dt;
    std::map<Point_pt, int> point_index_map;

    for (size_t j = 0; j < n_target; j++) {
        Point_pt p(target_ptr[2*j], target_ptr[2*j + 1]);
        dt.insert(p);
        point_index_map[p] = static_cast<int>(j);
    }
    for (size_t i = 0; i < n_query; i++) {
        Point_pt query(query_ptr[2*i], query_ptr[2*i + 1]);
        auto nearest_vertex = dt.nearest_vertex(query);
        double dx = query_ptr[2*i] - nearest_vertex->point().x();
        double dy = query_ptr[2*i + 1] - nearest_vertex->point().y();
        double distance = std::sqrt(dx*dx + dy*dy);
        indices[i] = point_index_map[nearest_vertex->point()];
        distances[i] = distance;
    }
    return py::make_tuple(py::cast(indices), py::cast(distances));
}

py::list find_clusters_cgal(
     py::array_t<double, py::array::c_style | py::array::forcecast> points,
     double threshold
 ) {
     auto points_buf = points.request();
     if (points_buf.ndim != 2 || points_buf.shape[1] != 2) {
         throw std::runtime_error("points must be Nx2 array");
     }
     size_t n_points = static_cast<size_t>(points_buf.shape[0]);
     if (n_points == 0) return py::list();
     const double* points_ptr = static_cast<const double*>(points_buf.ptr);

     DT dt;
     std::map<Point_pt, int> point_index_map;
     for (size_t i = 0; i < n_points; i++) {
         Point_pt p(points_ptr[2*i], points_ptr[2*i + 1]);
         dt.insert(p);
         point_index_map[p] = static_cast<int>(i);
     }

     std::vector<int> parent(n_points);
     for (size_t i = 0; i < n_points; i++) parent[i] = static_cast<int>(i);

     std::function<int(int)> find;
     find = [&parent, &find](int x) -> int {
         if (parent[x] != x) parent[x] = find(parent[x]);
         return parent[x];
     };
     auto union_sets = [&find, &parent](int x, int y) {
         int px = find(x); int py = find(y);
         if (px != py) parent[px] = py;
     };

    double threshold_sq = threshold * threshold;
    for (auto it = dt.finite_edges_begin(); it != dt.finite_edges_end(); ++it) {
        auto face = it->first;
        int index = it->second;
        
        auto v1 = face->vertex((index + 1) % 3);
        auto v2 = face->vertex((index + 2) % 3);

        if (dt.is_infinite(v1) || dt.is_infinite(v2)) continue;

         double dist_sq = CGAL::squared_distance(v1->point(), v2->point());
         if (dist_sq <= threshold_sq) {
             union_sets(point_index_map[v1->point()], point_index_map[v2->point()]);
         }
     }

     std::map<int, std::vector<int>> clusters;
     for (size_t i = 0; i < n_points; i++) {
         clusters[find(static_cast<int>(i))].push_back(static_cast<int>(i));
     }

     py::list result;
     for (const auto& cluster : clusters) {
         result.append(py::cast(cluster.second));
     }
     return result;
 }

py::tuple match_segment(
    py::array_t<double, py::array::c_style | py::array::forcecast> query_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments
) {
    auto query_buf = query_points.request();
    auto segments_buf = segments.request();
    if (query_buf.ndim != 2 || query_buf.shape[1] != 2) {
        throw std::runtime_error("query_points must be Nx2 array");
    }
    if (segments_buf.ndim != 2 || segments_buf.shape[1] != 4) {
        throw std::runtime_error("segments must be Mx4 array");
    }
    size_t n_query = static_cast<size_t>(query_buf.shape[0]);
    size_t n_segments = static_cast<size_t>(segments_buf.shape[0]);
    const double* query_ptr = static_cast<const double*>(query_buf.ptr);
    const double* segments_ptr = static_cast<const double*>(segments_buf.ptr);

    std::vector<Segment_k> segment_geoms;
    segment_geoms.reserve(n_segments);
    for (size_t j = 0; j < n_segments; j++) {
        Point_pt p1(segments_ptr[4*j],     segments_ptr[4*j + 1]);
        Point_pt p2(segments_ptr[4*j + 2], segments_ptr[4*j + 3]);
        segment_geoms.emplace_back(p1, p2);
    }

    AABB_Tree tree(segment_geoms.begin(), segment_geoms.end());
    tree.accelerate_distance_queries();

	std::vector<int> indices(n_query);
    std::vector<double> distances(n_query);

    for (size_t i = 0; i < n_query; i++) {
        Point_pt query(query_ptr[2*i], query_ptr[2*i + 1]);
        
        auto closest_pair = tree.closest_point_and_primitive(query);
        Point_pt closest_point = closest_pair.first;
        auto closest_primitive_it = closest_pair.second;
        double dist = std::sqrt(CGAL::to_double(CGAL::squared_distance(query, closest_point)));
        
        int best_idx = static_cast<int>(closest_primitive_it - segment_geoms.begin());
        
        indices[i] = best_idx;
        distances[i] = dist;
    }

    return py::make_tuple(py::cast(indices), py::cast(distances));
}
