/*
 * ACJ Core - CGAL-based spatial indexing for point-to-graph assignment
 *
 * This module provides high-performance spatial queries using CGAL
 * (Computational Geometry Algorithms Library). It wraps CGAL's Delaunay
 * triangulation for efficient nearest-neighbor queries.
 *
 * Functions:
 *   - match_point:     Find nearest point in target set for each query point
 *   - match_segment:   Find nearest line segment for each point
 */

 #include <cmath>
 #include <limits>
 #include <vector>
 #include <map>
 #include <functional>
 #include <utility>
 #include <tuple>
 #include <set>
 #include <deque>
 #include <algorithm>
 #include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Segment_Delaunay_graph_filtered_traits_2.h>
#include <CGAL/Segment_Delaunay_graph_2.h>

// CGAL type definitions
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K>                   DT;
typedef K::Point_2                                          Point_pt;

typedef CGAL::Simple_cartesian<double>                      CK;
typedef CGAL::Segment_Delaunay_graph_filtered_traits_2<
    CK, CGAL::Field_with_sqrt_tag>                          Gt;
typedef CGAL::Segment_Delaunay_graph_2<Gt>                  SDG2;
typedef Gt::Point_2                                         Point_sg;
typedef Gt::Site_2                                          Site_sg;

namespace py = pybind11;


// Mapea un ID de nodo (long) a su coordenada (Point_pt)
typedef std::map<long, Point_pt> NodeCoordMap;
// Mapea un ID de nodo a su grado (int)
typedef std::map<long, int> NodeDegreeMap;
// Mapea un ID de nodo a una lista de sus vecinos (IDs)
typedef std::map<long, std::vector<long>> AdjacencyMap;

/*
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
py::tuple match_point(
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
    std::map<Point_pt, int> point_index_map;

    // Insert all target points into the triangulation
    for (size_t j = 0; j < n_target; j++) {
        Point_pt p(target_ptr[2*j], target_ptr[2*j + 1]);
        dt.insert(p);
        point_index_map[p] = static_cast<int>(j);
    }

    // Query nearest neighbor for each query point
    for (size_t i = 0; i < n_query; i++) {
        Point_pt query(query_ptr[2*i], query_ptr[2*i + 1]);

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


/*
 * Encuentra clústeres de puntos dentro de un umbral usando CGAL.
 *
 * Esta versión es mucho más rápida (O(N log N)) que la O(N^2)
 * porque solo comprueba las aristas de la triangulación de Delaunay
 */
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
         auto v1 = face->vertex(CGAL::cw(index));
         auto v2 = face->vertex(CGAL::ccw(index));

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


/*
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
 */
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

     std::vector<long> indices(n_query);
     std::vector<double> distances(n_query);

     SDG2 sdg;
     std::map<Site_sg, long> segment_index_map;

     for (size_t j = 0; j < n_segments; j++) {
         Point_sg p1(segments_ptr[4*j], segments_ptr[4*j + 1]);
         Point_sg p2(segments_ptr[4*j + 2], segments_ptr[4*j + 3]);
         Site_sg site = Gt::Site_2::construct_segment(p1, p2);
         sdg.insert(site);
         segment_index_map[site] = static_cast<long>(j);
     }
     for (size_t i = 0; i < n_query; i++) {
         Point_sg query(query_ptr[2*i], query_ptr[2*i + 1]);
         auto nearest_neighbor_it = sdg.nearest_neighbor(query);

         double sq_dist = CGAL::squared_distance(query, nearest_neighbor_it->point());

         indices[i] = segment_index_map[nearest_neighbor_it->site()];
         distances[i] = std::sqrt(sq_dist);
     }

     return py::make_tuple(py::cast(indices), py::cast(distances));
 }


 // Traza un nuevo conjunto de segmentos conectando solo intersecciones.
 // Usado por ambas simplificaciones (topológica y geométrica).
 // Devuelve un conjunto de pares de IDs de nodo (new_start_id, new_end_id).
 std::set<std::pair<long, long>> trace_new_segments(
     const std::set<long>& intersection_ids,
     const AdjacencyMap& adjacency,
     const std::map<long, long>& node_to_new_node_map // Mapea ID original a nuevo ID
 ) {
     std::set<std::pair<long, long>> new_segments_set;
     std::set<std::pair<long, long>> visited_edges; // Evita trazar (a,b) y (b,a)

     for (long start_intersection_id : intersection_ids) {
         long start_new_node_id = node_to_new_node_map.at(start_intersection_id);

         std::deque<long> q;
         std::set<long> visited_nodes_in_path;

         q.push_back(start_intersection_id);
         visited_nodes_in_path.insert(start_intersection_id);

         while (!q.empty()) {
             long current_node_id = q.front();
             q.pop_front();

             if (!adjacency.count(current_node_id)) continue; // Nodo aislado

             for (long neighbor_id : adjacency.at(current_node_id)) {

                 // 1. Evitar volver atrás en el camino actual
                 if (visited_nodes_in_path.count(neighbor_id)) continue;

                 // 2. Evitar trazar un segmento que ya se trazó desde la otra dirección
                 std::pair<long, long> edge_key = std::minmax(current_node_id, neighbor_id);
                 if (visited_edges.count(edge_key)) continue;

                 // Si el vecino es OTRA intersección
                 if (intersection_ids.count(neighbor_id)) {
                     long end_new_node_id = node_to_new_node_map.at(neighbor_id);

                     // Añadir el nuevo segmento si conecta dos clústeres/nodos diferentes
                     if (start_new_node_id != end_new_node_id) {
                         new_segments_set.insert(std::minmax(start_new_node_id, end_new_node_id));
                     }
                     // Marcar el segmento original como visitado
                     visited_edges.insert(edge_key);
                 }
                 // Si el vecino es un nodo de grado 2 (camino)
                 else {
                     visited_nodes_in_path.insert(neighbor_id);
                     q.push_back(neighbor_id);
                     // Marcar el segmento original como visitado
                     visited_edges.insert(edge_key);
                 }
             }
         }
     }
     return new_segments_set;
 }

 // Construye las estructuras de grafo base desde los arrays de NumPy
 void build_graph_structures(
     py::array_t<double> nodes,   // (node_id, x, y)
     py::array_t<double> segments,// (segment_id, node_start, node_end)
     NodeCoordMap& node_map,
     NodeDegreeMap& node_degrees,
     AdjacencyMap& adjacency
 ) {
     auto nodes_buf = nodes.request();
     auto segments_buf = segments.request();
     const double* nodes_ptr = static_cast<const double*>(nodes_buf.ptr);
     const double* segments_ptr = static_cast<const double*>(segments_buf.ptr);

     size_t n_nodes = static_cast<size_t>(nodes_buf.shape[0]);
     size_t n_segments = static_cast<size_t>(segments_buf.shape[0]);

     // 1. Cargar todos los nodos y sus coordenadas
     for (size_t i = 0; i < n_nodes; i++) {
         long node_id = static_cast<long>(nodes_ptr[i * 3 + 0]);
         double x = nodes_ptr[i * 3 + 1];
         double y = nodes_ptr[i * 3 + 2];
         node_map[node_id] = Point_pt(x, y);
         node_degrees[node_id] = 0; // Inicializar grado
     }

     // 2. Construir adyacencia y calcular grados
     for (size_t j = 0; j < n_segments; j++) {
         long start_id = static_cast<long>(segments_ptr[j * 3 + 1]);
         long end_id = static_cast<long>(segments_ptr[j * 3 + 2]);

         adjacency[start_id].push_back(end_id);
         adjacency[end_id].push_back(start_id);

         node_degrees[start_id]++;
         node_degrees[end_id]++;
     }
 }

 /*
  * Simplifica un grafo eliminando nodos de grado 2.
  * Devuelve (new_nodes, new_segments)
  */
 py::tuple simplify_graph_topological_cgal(
     py::array_t<double> nodes,   // (node_id, x, y)
     py::array_t<double> segments // (segment_id, node_start, node_end)
 ) {
     NodeCoordMap node_map;
     NodeDegreeMap node_degrees;
     AdjacencyMap adjacency;

     build_graph_structures(nodes, segments, node_map, node_degrees, adjacency);

     std::set<long> intersection_ids;
     std::map<long, long> node_to_new_node_map; // Mapeo simple 1:1
     std::vector<std::tuple<long, double, double>> new_nodes_list;

     // 1. Identificar intersecciones (nodos != grado 2)
     for (const auto& pair : node_degrees) {
         if (pair.second != 2) {
             long node_id = pair.first;
             intersection_ids.insert(node_id);
             node_to_new_node_map[node_id] = node_id; // Mapea a sí mismo

             // Añadir a la lista de nuevos nodos
             Point_pt p = node_map[node_id];
             new_nodes_list.emplace_back(node_id, p.x(), p.y());
         }
     }

     // 2. Trazar nuevos segmentos
     std::set<std::pair<long, long>> new_segments_set =
         trace_new_segments(intersection_ids, adjacency, node_to_new_node_map);

     // 3. Formatear salida
     std::vector<std::tuple<long, long, long, double, double, double, double>> new_segments_list;
     long new_segment_id = 0;

     for (const auto& seg_pair : new_segments_set) {
         long id1 = seg_pair.first;
         long id2 = seg_pair.second;
         Point_pt p1 = node_map[id1];
         Point_pt p2 = node_map[id2];
         new_segments_list.emplace_back(new_segment_id++, id1, id2, p1.x(), p1.y(), p2.x(), p2.y());
     }

     return py::make_tuple(py::cast(new_nodes_list), py::cast(new_segments_list));
 }


 // --- Simplificación Geométrica en C++ ---
 /*
  * Simplifica un grafo fusionando intersecciones cercanas.
  * Devuelve (new_nodes, new_segments)
  */
 py::tuple simplify_graph_geometric_cgal(
     py::array_t<double> nodes,   // (node_id, x, y)
     py::array_t<double> segments,// (segment_id, node_start, node_end)
     double threshold
 ) {
     NodeCoordMap node_map;
     NodeDegreeMap node_degrees;
     AdjacencyMap adjacency;

     build_graph_structures(nodes, segments, node_map, node_degrees, adjacency);

     std::set<long> intersection_ids;
     std::vector<long> intersection_id_vec; // Para mapeo de índice
     std::vector<Point_pt> intersection_points;

     // 1. Identificar intersecciones (nodos != grado 2)
     for (const auto& pair : node_degrees) {
         if (pair.second != 2) {
             long node_id = pair.first;
             intersection_ids.insert(node_id);
             intersection_id_vec.push_back(node_id);
             intersection_points.push_back(node_map[node_id]);
         }
     }

     if (intersection_ids.empty()) { // No hay intersecciones
         return simplify_graph_topological_cgal(nodes, segments);
     }

     // 2. Agrupar intersecciones (Clustering)
     DT dt;
     std::map<Point_pt, int> point_to_idx_map;
     for (size_t i = 0; i < intersection_points.size(); i++) {
         dt.insert(intersection_points[i]);
         point_to_idx_map[intersection_points[i]] = static_cast<int>(i);
     }

     std::vector<int> parent(intersection_points.size());
     for (size_t i = 0; i < parent.size(); i++) parent[i] = static_cast<int>(i);

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
         auto v1 = it->first->vertex(CGAL::cw(it->second));
         auto v2 = it->first->vertex(CGAL::ccw(it->second));
         if (dt.is_infinite(v1) || dt.is_infinite(v2)) continue;
         if (CGAL::squared_distance(v1->point(), v2->point()) <= threshold_sq) {
             union_sets(point_to_idx_map[v1->point()], point_to_idx_map[v2->point()]);
         }
     }

     // 3. Crear nuevos nodos (centroides) y mapeo
     std::map<int, std::vector<int>> clusters; // root_parent -> [idx1, idx2, ...]
     for (size_t i = 0; i < parent.size(); i++) {
         clusters[find(static_cast<int>(i))].push_back(static_cast<int>(i));
     }

     std::vector<std::tuple<long, double, double>> new_nodes_list;
     std::map<long, long> node_to_new_node_map; // original_id -> new_cluster_id
     std::map<long, Point_pt> new_node_coords; // new_cluster_id -> Point_pt
     long new_cluster_id = 0;

     for (const auto& cluster : clusters) {
         double total_x = 0, total_y = 0;
         int count = 0;
         for (int idx : cluster.second) {
             long original_node_id = intersection_id_vec[idx];
             node_to_new_node_map[original_node_id] = new_cluster_id;

             Point_pt p = intersection_points[idx];
             total_x += p.x();
             total_y += p.y();
             count++;
         }
         double centroid_x = total_x / count;
         double centroid_y = total_y / count;

         new_nodes_list.emplace_back(new_cluster_id, centroid_x, centroid_y);
         new_node_coords[new_cluster_id] = Point_pt(centroid_x, centroid_y);
         new_cluster_id++;
     }

     // 4. Trazar nuevos segmentos
     std::set<std::pair<long, long>> new_segments_set =
         trace_new_segments(intersection_ids, adjacency, node_to_new_node_map);

     // 5. Formatear salida
     std::vector<std::tuple<long, long, long, double, double, double, double>> new_segments_list;
     long new_segment_id = 0;

     for (const auto& seg_pair : new_segments_set) {
         long id1 = seg_pair.first;
         long id2 = seg_pair.second;
         Point_pt p1 = new_node_coords[id1];
         Point_pt p2 = new_node_coords[id2];
         new_segments_list.emplace_back(new_segment_id++, id1, id2, p1.x(), p1.y(), p2.x(), p2.y());
     }

     return py::make_tuple(py::cast(new_nodes_list), py::cast(new_segments_list));
 }

 /*
  * Simplifica un grafo fusionando segmentos paralelos cercanos (e.g., doble calzada).
  * * Estrategia Óptima con CGAL:
  * 1. Usar un Árbol AABB para consultas rápidas de proximidad Segmento-Segmento.
  * 2. Comparar segmentos por:
  * a) Proximidad de sus cajas delimitadoras (Bounding Boxes).
  * b) Orientación (producto punto de vectores).
  * 3. Fusionar pares en un nuevo nodo/segmento central.
  */
 py::tuple simplify_graph_parallel_cgal(
      py::array_t<double> nodes,    // (node_id, x, y)
      py::array_t<double> segments,// (segment_id, node_start, node_end)
      double distance_threshold,
      double angle_threshold_deg // Umbral de ángulo para considerar paralelismo (e.g., 30 grados)
 ) {
      // -----------------------------------------------------------
      // PASO 1: Construir estructuras de grafo y lista de segmentos
      // -----------------------------------------------------------
      NodeCoordMap node_map;
      NodeDegreeMap node_degrees;
      AdjacencyMap adjacency;
      std::vector<SegmentInfo> segments_info_list;

      build_graph_structures(nodes, segments, node_map, node_degrees, adjacency, segments_info_list);

      // Extraer solo la geometría CGAL para el AABB Tree
      std::vector<Segment_k> segment_geometries;
      for (const auto& info : segments_info_list) {
          segment_geometries.push_back(std::get<5>(info));
      }

      if (segment_geometries.empty()) {
          return py::make_tuple(py::cast(std::vector<std::tuple<long, double, double>>{}), py::cast(std::vector<std::tuple<long, long, long, double, double, double, double>>{}));
      }

      // -----------------------------------------------------------
      // PASO 2: Construir el Árbol AABB para Búsqueda Rápida
      // -----------------------------------------------------------
      AABB_tree tree(segment_geometries.begin(), segment_geometries.end());
      tree.accelerate_distance_queries();

      // -----------------------------------------------------------
      // PASO 3: Identificar pares paralelos y realizar fusión (DSU)
      // -----------------------------------------------------------
      std::vector<int> parent(segments_info_list.size());
      for (size_t i = 0; i < parent.size(); i++) parent[i] = static_cast<int>(i);

      std::function<int(int)> find;
      find = [&parent, &find](int x) -> int {
          if (parent[x] != x) parent[x] = find(parent[x]);
          return parent[x];
      };
      auto union_sets = [&find, &parent](int x, int y) {
          int px = find(x); int py = find(y);
          if (px != py) parent[px] = py;
      };

      double threshold_sq = distance_threshold * distance_threshold;
      double angle_cos_threshold = std::cos(angle_threshold_deg * M_PI / 180.0);

      // Se usa un conjunto para evitar comparar el mismo par (i, j) dos veces
      std::set<std::pair<int, int>> checked_pairs;

      for (size_t i = 0; i < segments_info_list.size(); ++i) {
          const auto& seg_i_info = segments_info_list[i];
          const Segment_k& seg_i = std::get<5>(seg_i_info);
          Point_pt mid_i((seg_i.source().x() + seg_i.target().x()) / 2.0,
                         (seg_i.source().y() + seg_i.target().y()) / 2.0);
          Vector_k vec_i = seg_i.to_vector();

          // Consulta AABB: encontrar el punto más cercano en la estructura
          Point_pt nearest_to_mid = tree.closest_point(mid_i);

          // Si la distancia es mayor al umbral, no hay un segmento paralelo cerca.
          double dist_sq = CGAL::to_double(CGAL::squared_distance(mid_i, nearest_to_mid));
          if (dist_sq > threshold_sq) continue;

          // Iterar sobre todos los segmentos para encontrar un candidato que cumpla con el ángulo
          // (Esta es la parte menos óptima sin una estructura espacial avanzada para segmentos,
          // pero el AABB ya filtró los no-cercanos al midpoint).
          for (size_t j = i + 1; j < segments_info_list.size(); ++j) {
              if (i == j) continue;

              std::pair<int, int> pair_key = std::minmax(static_cast<int>(i), static_cast<int>(j));
              if (checked_pairs.count(pair_key)) continue;
              checked_pairs.insert(pair_key);

              const auto& seg_j_info = segments_info_list[j];
              const Segment_k& seg_j = std::get<5>(seg_j_info);

              // 3a. Proximidad (verificación final de endpoints)
              // Usamos la distancia cuadrada del extremo de i al segmento j.
              double dist_i_j = CGAL::to_double(tree.squared_distance(seg_i.source()));

              if (dist_i_j > threshold_sq) continue; // No están lo suficientemente cerca

              // 3b. Orientación (Paralelismo)
              Vector_k vec_j = seg_j.to_vector();
              double dot_product = CGAL::to_double(vec_i * vec_j);
              double magnitude_prod = CGAL::to_double(vec_i.squared_length() * vec_j.squared_length());
              double cos_angle = dot_product / std::sqrt(magnitude_prod);

              // Chequear si son paralelos (ángulo cercano a 0) O antiparalelos (ángulo cercano a 180)
              if (std::abs(cos_angle) > angle_cos_threshold) {
                  // Son paralelos o antiparalelos Y están cerca -> FUSIONAR
                  union_sets(static_cast<int>(i), static_cast<int>(j));
              }
          }
      }
      std::map<int, std::vector<int>> segment_clusters; // root_parent -> [seg_idx1, seg_idx2, ...]
            for (size_t i = 0; i < parent.size(); i++) {
                segment_clusters[find(static_cast<int>(i))].push_back(static_cast<int>(i));
            }

            std::vector<std::tuple<long, double, double>> new_nodes_list;
            std::map<long, long> node_to_new_node_map; // original_id -> new_cluster_id (Para el trazado)
            std::map<long, Point_pt> new_node_coords; // new_cluster_id -> Point_pt
            long new_cluster_id_counter = 0;
            long new_node_id_offset = 10000000; // Usar un ID muy grande para nuevos nodos

            // Mapeo de nodos originales a los IDs de nuevos nodos/centroides
            for (const auto& cluster_pair : segment_clusters) {
                const auto& segment_indices = cluster_pair.second;
                if (segment_indices.size() == 1) continue; // No fusionar segmentos solos

                // Para cada clúster de segmentos:
                // 1. Identificar todos los nodos originales involucrados
                std::set<long> original_node_ids;
                for (int idx : segment_indices) {
                    original_node_ids.insert(std::get<1>(segments_info_list[idx])); // Start ID
                    original_node_ids.insert(std::get<2>(segments_info_list[idx])); // End ID
                }

                // 2. Crear un nuevo nodo "centroide" que reemplace a todos los originales
                long new_id = new_node_id_offset + (new_cluster_id_counter++);

                // Calculamos el centroide de los NODOS originales del cluster
                double total_x = 0, total_y = 0;
                for (long original_id : original_node_ids) {
                    Point_pt p = node_map.at(original_id);
                    total_x += CGAL::to_double(p.x());
                    total_y += CGAL::to_double(p.y());
                }
                double centroid_x = total_x / original_node_ids.size();
                double centroid_y = total_y / original_node_ids.size();

                Point_pt centroid_p(centroid_x, centroid_y);

                // 3. Mapear cada nodo original al nuevo ID
                for (long original_id : original_node_ids) {
                    node_to_new_node_map[original_id] = new_id;
                }

                // 4. Registrar el nuevo nodo
                new_nodes_list.emplace_back(new_id, centroid_x, centroid_y);
                new_node_coords[new_id] = centroid_p;
            }

            // Añadir los nodos que NO fueron fusionados (nodos que eran intersecciones/extremos)
            // También se aplica la simplificación topológica aquí (eliminar grado 2)
            for (const auto& pair : node_degrees) {
                long node_id = pair.first;
                // Solo conservamos nodos que no son de grado 2 Y no fueron ya fusionados
                if (pair.second != 2 && node_to_new_node_map.find(node_id) == node_to_new_node_map.end()) {
                    node_to_new_node_map[node_id] = node_id; // Mapea a sí mismo
                    Point_pt p = node_map.at(node_id);
                    new_nodes_list.emplace_back(node_id, CGAL::to_double(p.x()), CGAL::to_double(p.y()));
                    new_node_coords[node_id] = p;
                } else if (pair.second == 2 && node_to_new_node_map.find(node_id) == node_to_new_node_map.end()) {
                    // Estos nodos de grado 2 serán saltados por trace_new_segments.
                    // NO se agregan a new_nodes_list.
                }
            }

            // -----------------------------------------------------------
            // PASO 5: Trazar nuevos segmentos (Conexiones)
            // -----------------------------------------------------------
            // Las "intersecciones" para el trazado son ahora todos los nodos finales que tienen un mapeo.
            std::set<long> final_intersections;
            for(const auto& pair : node_to_new_node_map) {
                final_intersections.insert(pair.first);
            }

            std::set<std::pair<long, long>> new_segments_set =
                trace_new_segments(final_intersections, adjacency, node_to_new_node_map);

            // -----------------------------------------------------------
            // PASO 6: Formatear salida
            // -----------------------------------------------------------
            std::vector<std::tuple<long, long, long, double, double, double, double>> final_segments_list;
            long final_segment_id = 0;

            for (const auto& seg_pair : new_segments_set) {
                long id1 = seg_pair.first;
                long id2 = seg_pair.second;

                if (new_node_coords.find(id1) == new_node_coords.end() || new_node_coords.find(id2) == new_node_coords.end()) {
                    // Debería ser imposible si el trazado es correcto, pero es un chequeo de seguridad
                    continue;
                }

                Point_pt p1 = new_node_coords.at(id1);
                Point_pt p2 = new_node_coords.at(id2);
                final_segments_list.emplace_back(
                    final_segment_id++,
                    id1,
                    id2,
                    CGAL::to_double(p1.x()),
                    CGAL::to_double(p1.y()),
                    CGAL::to_double(p2.x()),
                    CGAL::to_double(p2.y())
                );
            }

            return py::make_tuple(py::cast(new_nodes_list), py::cast(final_segments_list));
}


/*
 * Python module definition for acj_core.
 *
 * This module is imported by the Python MapIndex class to perform
 * high-performance spatial queries.
 */


 // --- Definición del Módulo de Python ---
 PYBIND11_MODULE(acj_core, m) {
     m.doc() = "ACJ Core - CGAL-based spatial indexing & graph simplification";

     m.def("match_point", &match_point,
           "Encuentra el punto objetivo más cercano para cada punto de consulta.",
           py::arg("query_points"), py::arg("target_points"));

     m.def("find_clusters_cgal", &find_clusters_cgal,
           "Encuentra clústeres de puntos dentro de un umbral (O(N log N)).",
           py::arg("points"), py::arg("threshold"));

     m.def("match_segment", &match_segment,
           "Encuentra el segmento de línea más cercano para cada punto de consulta.",
           py::arg("query_points"), py::arg("segments"));

     // --- NUEVAS FUNCIONES EXPUESTAS ---
     m.def("simplify_graph_topological_cgal", &simplify_graph_topological_cgal,
           "Simplifica un grafo eliminando nodos de grado 2 (C++).",
           py::arg("nodes"), py::arg("segments"));

     m.def("simplify_graph_geometric_cgal", &simplify_graph_geometric_cgal,
           "Simplifica un grafo fusionando intersecciones cercanas (C++).",
           py::arg("nodes"), py::arg("segments"), py::arg("threshold"));

     m.def("simplify_graph_parallel_cgal", &simplify_graph_parallel_cgal,
            "Simplifica un grafo fusionando segmentos paralelos (e.g., doble calzada) (C++).",
            py::arg("nodes"), py::arg("segments"), py::arg("distance_threshold"), py::arg("angle_threshold_deg"));
 }
