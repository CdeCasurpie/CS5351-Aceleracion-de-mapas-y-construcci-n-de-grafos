#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "graph_simplify.hpp"
// ================= Helpers Internos =================

void douglas_peucker(const std::vector<Point_pt>& points, size_t start, size_t end, double epsilon, std::vector<bool>& keep) {
    if (start + 1 >= end) return;

    double max_dist_sq = 0.0;
    size_t index = start;
    Segment_k seg(points[start], points[end]);

    for (size_t i = start + 1; i < end; ++i) {
        double dist_sq = CGAL::to_double(CGAL::squared_distance(points[i], seg));
        if (dist_sq > max_dist_sq) {
            max_dist_sq = dist_sq;
            index = i;
        }
    }

    if (max_dist_sq > epsilon * epsilon) {
        keep[index] = true;
        douglas_peucker(points, start, index, epsilon, keep);
        douglas_peucker(points, index, end, epsilon, keep);
    }
}

std::vector<Point_pt> simplify_polyline(const std::vector<Point_pt>& points, double epsilon) {
    if (points.size() <= 2) return points;
    std::vector<bool> keep(points.size(), false);
    keep[0] = true;
    keep[points.size() - 1] = true;

    douglas_peucker(points, 0, points.size() - 1, epsilon, keep);

    std::vector<Point_pt> result;
    for (size_t i = 0; i < points.size(); ++i) {
        if (keep[i]) result.push_back(points[i]);
    }
    return result;
}
// ================== ================= Revisar esta parte ================= ================ 

Polygon_exact create_segment_buffer(double x1, double y1, double x2, double y2, double radius) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double length = std::sqrt(dx * dx + dy * dy);
    
    if (length == 0.0) return Polygon_exact();
    
    double nx = (-dy / length) * radius;
    double ny = (dx / length) * radius;
    
    Polygon_exact poly;
    poly.push_back(Exact_K::Point_2(x1 + nx, y1 + ny));
    poly.push_back(Exact_K::Point_2(x2 + nx, y2 + ny));
    poly.push_back(Exact_K::Point_2(x2 - nx, y2 - ny));
    poly.push_back(Exact_K::Point_2(x1 - nx, y1 - ny));
    
    if (poly.is_clockwise_oriented()) {
        poly.reverse_orientation();
    }
    return poly;
}

Polygon_with_holes_fast convert_to_epic(const Polygon_with_holes_exact& exact_poly) {
    Polygon_fast outer;
    for (auto v = exact_poly.outer_boundary().vertices_begin(); v != exact_poly.outer_boundary().vertices_end(); ++v) {
        outer.push_back(Point_pt(CGAL::to_double(v->x()), CGAL::to_double(v->y())));
    }
    std::vector<Polygon_fast> holes;
    for (auto h = exact_poly.holes_begin(); h != exact_poly.holes_end(); ++h) {
        Polygon_fast hole;
        for (auto v = h->vertices_begin(); v != h->vertices_end(); ++v) {
            hole.push_back(Point_pt(CGAL::to_double(v->x()), CGAL::to_double(v->y())));
        }
        holes.push_back(hole);
    }
    return Polygon_with_holes_fast(outer, holes.begin(), holes.end());
}
// ================== ================= Revisar esta parte ================= ================ 
std::set<std::pair<long, long>> trace_new_segments(
    const std::set<long>& intersection_ids,
    const AdjacencyMap& adjacency,
    const std::map<long, long>& node_to_new_node_map
) {
    std::set<std::pair<long, long>> new_segments_set;
    std::set<std::pair<long, long>> visited_edges; 

    for (long start_intersection_id : intersection_ids) {
        if (node_to_new_node_map.find(start_intersection_id) == node_to_new_node_map.end()) continue;
        long start_new_node_id = node_to_new_node_map.at(start_intersection_id);

        std::deque<long> q;
        std::set<long> visited_nodes_in_path;
        q.push_back(start_intersection_id);
        visited_nodes_in_path.insert(start_intersection_id);

        while (!q.empty()) {
            long current_node_id = q.front();
            q.pop_front();
            if (!adjacency.count(current_node_id)) continue;

            for (long neighbor_id : adjacency.at(current_node_id)) {
                if (visited_nodes_in_path.count(neighbor_id)) continue;
                std::pair<long, long> edge_key = std::minmax(current_node_id, neighbor_id);
                if (visited_edges.count(edge_key)) continue;

                if (intersection_ids.count(neighbor_id)) {
                    if (node_to_new_node_map.find(neighbor_id) == node_to_new_node_map.end()) continue;
                    long end_new_node_id = node_to_new_node_map.at(neighbor_id);

                    if (start_new_node_id != end_new_node_id) {
                        new_segments_set.insert(std::minmax(start_new_node_id, end_new_node_id));
                    }
                    visited_edges.insert(edge_key);
                } else {
                    visited_nodes_in_path.insert(neighbor_id);
                    q.push_back(neighbor_id);
                    visited_edges.insert(edge_key);
                }
            }
        }
    }
    return new_segments_set;
}

void build_graph_structures(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
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

    for (size_t i = 0; i < n_nodes; i++) {
        long node_id = static_cast<long>(nodes_ptr[i * 3 + 0]);
        double x = nodes_ptr[i * 3 + 1];
        double y = nodes_ptr[i * 3 + 2];
        node_map[node_id] = Point_pt(x, y);
        node_degrees[node_id] = 0;
    }
    for (size_t j = 0; j < n_segments; j++) {
        long start_id = static_cast<long>(segments_ptr[j * 3 + 1]);
        long end_id = static_cast<long>(segments_ptr[j * 3 + 2]);
        
        if (node_map.find(start_id) == node_map.end() || node_map.find(end_id) == node_map.end()) {
            continue; 
        }

        adjacency[start_id].push_back(end_id);
        adjacency[end_id].push_back(start_id);
        node_degrees[start_id]++;
        node_degrees[end_id]++;
    }
}

void build_graph_structures(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    NodeCoordMap& node_map,
    NodeDegreeMap& node_degrees,
    AdjacencyMap& adjacency,
    std::vector<SegmentInfo>& segments_info_list
) {
    build_graph_structures(nodes, segments, node_map, node_degrees, adjacency);
    
    auto segments_buf = segments.request();
    const double* segments_ptr = static_cast<const double*>(segments_buf.ptr);
    size_t n_segments = static_cast<size_t>(segments_buf.shape[0]);

    for (size_t j = 0; j < n_segments; j++) {
        long seg_id = static_cast<long>(segments_ptr[j * 3 + 0]);
        long start_id = static_cast<long>(segments_ptr[j * 3 + 1]);
        long end_id = static_cast<long>(segments_ptr[j * 3 + 2]);

        if (node_map.find(start_id) == node_map.end() || node_map.find(end_id) == node_map.end()) {
            continue;
        }

        Point_pt p1 = node_map.at(start_id);
        Point_pt p2 = node_map.at(end_id);
        Segment_k seg_geom(p1, p2);

        segments_info_list.emplace_back(seg_id, start_id, end_id, p1, p2, seg_geom);
    }
}

SimplificationResult minkowski_guided_simplify(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes_x_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes_y_in,
    py::array_t<long> seg_start_in, py::array_t<long> seg_end_in, double radius)
{
    auto nx = nodes_x_in.unchecked<1>();
    auto ny = nodes_y_in.unchecked<1>();
    auto ss = seg_start_in.unchecked<1>();
    auto se = seg_end_in.unchecked<1>();
    
    size_t num_nodes = nx.shape(0);
    size_t num_edges = ss.shape(0);

    std::vector<int> degree(num_nodes, 0);
    std::map<int, std::vector<int>> original_adj;
    
    for (size_t i = 0; i < num_edges; ++i) {
        long u = ss(i);
        long v = se(i);
        degree[u]++;
        degree[v]++;
        original_adj[u].push_back(v);
        original_adj[v].push_back(u);
    }

    std::vector<int> anchors;
    for (size_t i = 0; i < num_nodes; ++i) {
        //if (degree[i] != 2) anchors.push_back(i);
		anchors.push_back(i);
    }

    Polygon_set_exact city_polygon_set;
    for (size_t i = 0; i < num_edges; ++i) {
        long u = ss(i);
        long v = se(i);
        Polygon_exact buffer = create_segment_buffer(nx(u), ny(u), nx(v), ny(v), radius);
        if (!buffer.is_empty()) {
            city_polygon_set.join(buffer);
        }
    }

    std::vector<Polygon_with_holes_exact> res;
    city_polygon_set.polygons_with_holes(std::back_inserter(res));
    if (res.empty()) return SimplificationResult{py::make_tuple(py::array(), py::array(), py::array(), py::array())};

    std::vector<Point_pt> skel_points;
    std::map<int, std::vector<int>> skel_adj;
    int skel_id_counter = 0;

    for (const auto& exact_poly : res) {
        Polygon_with_holes_fast fast_poly = convert_to_epic(exact_poly);
        
        auto skeleton = CGAL::create_interior_straight_skeleton_2(
            fast_poly.outer_boundary().vertices_begin(),
            fast_poly.outer_boundary().vertices_end(),
            fast_poly.holes_begin(),
            fast_poly.holes_end()
        );

        std::map<Vertex_const_handle, int> skel_vertex_to_id;
        for (auto v = skeleton->vertices_begin(); v != skeleton->vertices_end(); ++v) {
            skel_points.push_back(v->point());
            skel_vertex_to_id[v] = skel_id_counter++;
        }

        for (auto edge = skeleton->halfedges_begin(); edge != skeleton->halfedges_end(); ++edge) {
            if (edge->is_inner_bisector()) {
                int u = skel_vertex_to_id[edge->vertex()];
                int v = skel_vertex_to_id[edge->opposite()->vertex()];
                skel_adj[u].push_back(v);
            }
        }
    }

    Tree tree;
    for (const auto& pt : skel_points) {
        tree.insert(pt);
    }

    std::map<int, int> anchor_to_skel_id;
    for (int anchor : anchors) {
        Point_pt query(nx(anchor), ny(anchor));
        Neighbor_search search(tree, query, 1);
        Point_pt nearest = search.begin()->first;
        
        for (size_t i = 0; i < skel_points.size(); ++i) {
            if (skel_points[i] == nearest) {
                anchor_to_skel_id[anchor] = i;
                break;
            }
        }
    }

    std::set<std::pair<int, int>> final_edges_set;

    // Conectar anclas usando BFS rastreando el grafo original
    for (int u : anchors) {
        for (int initial_neighbor : original_adj[u]) {
            
            int prev = u;
            int curr = initial_neighbor;
            
            while (degree[curr] == 2) {
                int next_node = (original_adj[curr][0] == prev) ? original_adj[curr][1] : original_adj[curr][0];
                prev = curr;
                curr = next_node;
            }
            
            int v = curr;

            if (u < v) { 
                int start_skel = anchor_to_skel_id[u];
                int end_skel = anchor_to_skel_id[v];

                std::queue<int> q;
                std::map<int, int> parent;
                q.push(start_skel);
                parent[start_skel] = -1;
                
                bool found = false;
                while (!q.empty()) {
                    int current_skel = q.front(); q.pop();
                    if (current_skel == end_skel) { found = true; break; }
                    
                    for (int neighbor : skel_adj[current_skel]) {
                        if (parent.find(neighbor) == parent.end()) {
                            parent[neighbor] = current_skel;
                            q.push(neighbor);
                        }
                    }
                }

                if (found) {
                    int current_skel = end_skel;
                    while (parent[current_skel] != -1) {
                        int p = parent[current_skel];
                        final_edges_set.insert({std::min(current_skel, p), std::max(current_skel, p)});
                        current_skel = p;
                    }
                }
            }
        }
    }

    std::vector<double> out_nx, out_ny;
    std::vector<long> out_ss, out_se;
    std::map<int, long> final_node_mapper;

    long new_id_counter = 0;
    for (auto edge : final_edges_set) {
        int u = edge.first;
        int v = edge.second;

        if (final_node_mapper.find(u) == final_node_mapper.end()) {
            final_node_mapper[u] = new_id_counter++;
            out_nx.push_back(CGAL::to_double(skel_points[u].x()));
            out_ny.push_back(CGAL::to_double(skel_points[u].y()));
        }
        if (final_node_mapper.find(v) == final_node_mapper.end()) {
            final_node_mapper[v] = new_id_counter++;
            out_nx.push_back(CGAL::to_double(skel_points[v].x()));
            out_ny.push_back(CGAL::to_double(skel_points[v].y()));
        }

        out_ss.push_back(final_node_mapper[u]);
        out_se.push_back(final_node_mapper[v]);
    }

    auto np_nx = py::array_t<double>(out_nx.size(), out_nx.data());
    auto np_ny = py::array_t<double>(out_ny.size(), out_ny.data());
    auto np_ss = py::array_t<long>(out_ss.size(), out_ss.data());
    auto np_se = py::array_t<long>(out_se.size(), out_se.data());

    return SimplificationResult{py::make_tuple(np_nx, np_ny, np_ss, np_se)};
}
SimplificationResult simplify_graph_topological_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments
) {
    NodeCoordMap node_map;
    NodeDegreeMap node_degrees;
    AdjacencyMap adjacency;
    build_graph_structures(nodes, segments, node_map, node_degrees, adjacency);

    std::set<long> intersection_ids;
    std::map<long, long> node_to_new_node_map;
    std::vector<std::tuple<long, double, double>> new_nodes_list;

    for (const auto& pair : node_degrees) {
        if (pair.second != 2) {
            long node_id = pair.first;
            intersection_ids.insert(node_id);
            node_to_new_node_map[node_id] = node_id;
            Point_pt p = node_map[node_id];
            new_nodes_list.emplace_back(node_id, p.x(), p.y());
        }
    }

    std::set<std::pair<long, long>> new_segments_set =
        trace_new_segments(intersection_ids, adjacency, node_to_new_node_map);

    std::vector<std::tuple<long, long, long, double, double, double, double>> new_segments_list;
    long new_segment_id = 0;
    for (const auto& seg_pair : new_segments_set) {
        long id1 = seg_pair.first;
        long id2 = seg_pair.second;
        Point_pt p1 = node_map[id1];
        Point_pt p2 = node_map[id2];
        new_segments_list.emplace_back(new_segment_id++, id1, id2, p1.x(), p1.y(), p2.x(), p2.y());
    }
    return SimplificationResult{py::make_tuple(py::cast(new_nodes_list), py::cast(new_segments_list))};
}

SimplificationResult simplify_graph_minkowski_cgal(
	py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
	py::array_t<double, py::array::c_style | py::array::forcecast> segments,
	double radius
){
	NodeCoordMap node_map;
	NodeDegreeMap node_degrees;
	AdjacencyMap adjacency;
	std::vector<SegmentInfo> segments_info_list;
	build_graph_structures(nodes, segments, node_map, node_degrees, adjacency, segments_info_list);
	Polygon_set_exact poly_set;
	for (const auto& seg_info : segments_info_list) {
        Point_pt p1 = std::get<3>(seg_info);
        Point_pt p2 = std::get<4>(seg_info);

        double dx = CGAL::to_double(p2.x() - p1.x());
        double dy = CGAL::to_double(p2.y() - p1.y());
        double len = std::sqrt(dx*dx + dy*dy);
        if (len == 0) continue;

        double nx = -dy / len * radius;
        double ny = dx / len * radius;

        Polygon_exact poly;
        poly.push_back(Exact_K::Point_2(CGAL::to_double(p1.x()) + nx, CGAL::to_double(p1.y()) + ny));
        poly.push_back(Exact_K::Point_2(CGAL::to_double(p2.x()) + nx, CGAL::to_double(p2.y()) + ny));
        poly.push_back(Exact_K::Point_2(CGAL::to_double(p2.x()) - nx, CGAL::to_double(p2.y()) - ny));
        poly.push_back(Exact_K::Point_2(CGAL::to_double(p1.x()) - nx, CGAL::to_double(p1.y()) - ny));

        if (poly.is_simple()) {
            if (poly.is_clockwise_oriented()) poly.reverse_orientation();
            poly_set.join(poly);
        }
    }

	std::vector<Polygon_with_holes_exact> res_polygons;
    poly_set.polygons_with_holes(std::back_inserter(res_polygons));
	std::vector<std::tuple<long, double, double>> new_nodes_list;
    std::vector<std::tuple<long, long, long, double, double, double, double>> new_segments_list;

    long node_id_counter = 0;
    long segment_id_counter = 0;
    std::map<Exact_K::Point_2, long> exact_node_map;

	for (const auto& pwh : res_polygons) {
        Exact_K exact_kernel;
		Straight_skeleton_ptr ss = CGAL::create_interior_straight_skeleton_2(
            pwh.outer_boundary().vertices_begin(),
            pwh.outer_boundary().vertices_end(),
            pwh.holes_begin(),
            pwh.holes_end(),
			exact_kernel
        );
        if (!ss) continue;

        for (auto edge = ss->halfedges_begin(); edge != ss->halfedges_end(); ++edge)
		{
            if (!edge->is_bisector() || !edge->is_inner_bisector()) continue; 
            
            auto v1 = edge->vertex();
            auto v2 = edge->opposite()->vertex();
            
            // Solo nos quedamos con el eje central puro, descartando las diagonales a las esquinas
            if (!v1->is_skeleton() || !v2->is_skeleton()) continue;

            Exact_K::Point_2 p1 = v1->point();
            Exact_K::Point_2 p2 = v2->point();

            long id1, id2;
            if (exact_node_map.find(p1) == exact_node_map.end()) {
                id1 = node_id_counter++;
                exact_node_map[p1] = id1;
                new_nodes_list.emplace_back(id1, CGAL::to_double(p1.x()), CGAL::to_double(p1.y()));
            } else id1 = exact_node_map[p1];

            if (exact_node_map.find(p2) == exact_node_map.end()) {
                id2 = node_id_counter++;
                exact_node_map[p2] = id2;
                new_nodes_list.emplace_back(id2, CGAL::to_double(p2.x()), CGAL::to_double(p2.y()));
            } else id2 = exact_node_map[p2];

            // Usamos NUESTROS ids generados para evitar aristas duplicadas
            if (id1 < id2) {
                new_segments_list.emplace_back(
                    segment_id_counter++, id1, id2,
                    CGAL::to_double(p1.x()), CGAL::to_double(p1.y()),
                    CGAL::to_double(p2.x()), CGAL::to_double(p2.y())
                );
				
        	}   
		}
    }
    
    return SimplificationResult{py::make_tuple(py::cast(new_nodes_list), py::cast(new_segments_list))};
}

SimplificationResult simplify_graph_geometric_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double threshold
) {
    NodeCoordMap node_map;
    NodeDegreeMap node_degrees;
    AdjacencyMap adjacency;
    build_graph_structures(nodes, segments, node_map, node_degrees, adjacency);

    std::set<long> intersection_ids;
    std::vector<long> intersection_id_vec;
    std::vector<Point_pt> intersection_points;

    for (const auto& pair : node_degrees) {
        if (pair.second != 2) {
            long node_id = pair.first;
            intersection_ids.insert(node_id);
            intersection_id_vec.push_back(node_id);
            intersection_points.push_back(node_map[node_id]);
        }
    }
    if (intersection_ids.empty()) {
        return simplify_graph_topological_cgal(nodes, segments);
    }

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
        auto v1 = it->first->vertex((it->second + 1) % 3);
        auto v2 = it->first->vertex((it->second + 2) % 3);
        if (dt.is_infinite(v1) || dt.is_infinite(v2)) continue;
        if (CGAL::squared_distance(v1->point(), v2->point()) <= threshold_sq) {
            union_sets(point_to_idx_map[v1->point()], point_to_idx_map[v2->point()]);
        }
    }

    std::map<int, std::vector<int>> clusters;
    for (size_t i = 0; i < parent.size(); i++) {
        clusters[find(static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    std::vector<std::tuple<long, double, double>> new_nodes_list;
    std::map<long, long> node_to_new_node_map;
    std::map<long, Point_pt> new_node_coords;
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

    std::set<std::pair<long, long>> new_segments_set =
        trace_new_segments(intersection_ids, adjacency, node_to_new_node_map);

    std::vector<std::tuple<long, long, long, double, double, double, double>> new_segments_list;
    long new_segment_id = 0;
    for (const auto& seg_pair : new_segments_set) {
        long id1 = seg_pair.first;
        long id2 = seg_pair.second;
        Point_pt p1 = new_node_coords[id1];
        Point_pt p2 = new_node_coords[id2];
        new_segments_list.emplace_back(new_segment_id++, id1, id2, p1.x(), p1.y(), p2.x(), p2.y());
    }
    return SimplificationResult{py::make_tuple(py::cast(new_nodes_list), py::cast(new_segments_list))};
}

SimplificationResult simplify_graph_parallel_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double distance_threshold,
    double angle_threshold_deg
) {
    NodeCoordMap node_map;
    NodeDegreeMap node_degrees;
    AdjacencyMap adjacency;
    std::vector<SegmentInfo> segments_info_list;

    build_graph_structures(nodes, segments, node_map, node_degrees, adjacency, segments_info_list);

    if (segments_info_list.empty()) {
        return SimplificationResult{py::make_tuple(py::cast(std::vector<std::tuple<long, double, double>>{}), py::cast(std::vector<std::tuple<long, long, long, double, double, double, double>>{}))};
    }

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
    double angle_cos_threshold = std::cos(angle_threshold_deg * acos(-1.0) / 180.0);

    for (size_t i = 0; i < segments_info_list.size(); ++i) {
        const auto& seg_i_info = segments_info_list[i];
        const Segment_k& seg_i = std::get<5>(seg_i_info);
        Vector_k vec_i = seg_i.to_vector();

        for (size_t j = i + 1; j < segments_info_list.size(); ++j) {
            const auto& seg_j_info = segments_info_list[j];
            const Segment_k& seg_j = std::get<5>(seg_j_info);

            double dist_sq = CGAL::to_double(CGAL::squared_distance(seg_i, seg_j));
            if (dist_sq > threshold_sq) continue;

            Vector_k vec_j = seg_j.to_vector();
            double dot_product = CGAL::to_double(vec_i * vec_j);
            double magnitude_prod = CGAL::to_double(vec_i.squared_length() * vec_j.squared_length());
            
            if (magnitude_prod == 0) continue;
            double cos_angle = dot_product / std::sqrt(magnitude_prod);

            if (std::abs(cos_angle) > angle_cos_threshold) {
                union_sets(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }
    
    std::map<int, std::vector<int>> segment_clusters;
    for (size_t i = 0; i < parent.size(); i++) {
        segment_clusters[find(static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    std::vector<std::tuple<long, double, double>> new_nodes_list;
    std::map<long, long> node_to_new_node_map;
    std::map<long, Point_pt> new_node_coords;
    long new_cluster_id_counter = 0;
    long new_node_id_offset = 10000000;

    for (const auto& cluster_pair : segment_clusters) {
        const auto& segment_indices = cluster_pair.second;
        if (segment_indices.size() == 1) continue; 

        std::set<long> original_node_ids;
        for (int idx : segment_indices) {
            original_node_ids.insert(std::get<1>(segments_info_list[idx])); // Start ID
            original_node_ids.insert(std::get<2>(segments_info_list[idx])); // End ID
        }

        long new_id = new_node_id_offset + (new_cluster_id_counter++);
        double total_x = 0, total_y = 0;
        for (long original_id : original_node_ids) {
            Point_pt p = node_map.at(original_id);
            total_x += CGAL::to_double(p.x());
            total_y += CGAL::to_double(p.y());
        }
        double centroid_x = total_x / original_node_ids.size();
        double centroid_y = total_y / original_node_ids.size();
        Point_pt centroid_p(centroid_x, centroid_y);

        for (long original_id : original_node_ids) {
            node_to_new_node_map[original_id] = new_id;
        }
        new_nodes_list.emplace_back(new_id, centroid_x, centroid_y);
        new_node_coords[new_id] = centroid_p;
    }

    for (const auto& pair : node_degrees) {
        long node_id = pair.first;
        if (pair.second != 2 && node_to_new_node_map.find(node_id) == node_to_new_node_map.end()) {
            node_to_new_node_map[node_id] = node_id;
            Point_pt p = node_map.at(node_id);
            new_nodes_list.emplace_back(node_id, CGAL::to_double(p.x()), CGAL::to_double(p.y()));
            new_node_coords[node_id] = p;
        }
    }

    std::set<long> final_intersections;
    for(const auto& pair : node_to_new_node_map) {
        final_intersections.insert(pair.first);
    }

    std::set<std::pair<long, long>> new_segments_set =
        trace_new_segments(final_intersections, adjacency, node_to_new_node_map);

    std::vector<std::tuple<long, long, long, double, double, double, double>> final_segments_list;
    long final_segment_id = 0;

    for (const auto& seg_pair : new_segments_set) {
        long id1 = seg_pair.first;
        long id2 = seg_pair.second;
        if (new_node_coords.find(id1) == new_node_coords.end() || new_node_coords.find(id2) == new_node_coords.end()) {
            continue;
        }
        Point_pt p1 = new_node_coords.at(id1);
        Point_pt p2 = new_node_coords.at(id2);
        final_segments_list.emplace_back(
            final_segment_id++,
            id1, id2,
            CGAL::to_double(p1.x()), CGAL::to_double(p1.y()),
            CGAL::to_double(p2.x()), CGAL::to_double(p2.y())
        );
    }
    return SimplificationResult{py::make_tuple(py::cast(new_nodes_list), py::cast(final_segments_list))};
}

// =====================================================================
// ACJ: IMPLEMENTACIÓN CORE
// =====================================================================

SimplificationResult simplify_graph_acj_master_cgal(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> segments,
    double angulo_maximo_desviacion,
    double factor_ajuste,
    double factor_epsilon,
    bool with_index
) {
    auto nodes_buf = nodes.request();
    auto segs_buf = segments.request();

    if (nodes_buf.ndim != 2 || nodes_buf.shape[1] != 3) throw std::runtime_error("Nodes must be Nx3 array [id, x, y]");
    if (segs_buf.ndim != 2 || segs_buf.shape[1] != 3) throw std::runtime_error("Segments must be Mx3 array [edge_id, u, v]");

    size_t num_nodes = static_cast<size_t>(nodes_buf.shape[0]);
    size_t num_segs  = static_cast<size_t>(segs_buf.shape[0]);

    const double* nodes_ptr = static_cast<const double*>(nodes_buf.ptr);
    const double* segs_ptr  = static_cast<const double*>(segs_buf.ptr);

    std::unordered_map<long long, Point_pt> node_coords;
    std::unordered_map<long long, int> in_degree;
    std::unordered_map<long long, int> out_degree;
    std::unordered_map<long long, std::vector<std::pair<long long, long long>>> adj_list;

    node_coords.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        long long id = static_cast<long long>(nodes_ptr[3 * i]);
        node_coords[id] = Point_pt(nodes_ptr[3 * i + 1], nodes_ptr[3 * i + 2]);
    }

    for (size_t i = 0; i < num_segs; ++i) {
        long long edge_id = static_cast<long long>(segs_ptr[3 * i]);
        long long u       = static_cast<long long>(segs_ptr[3 * i + 1]);
        long long v       = static_cast<long long>(segs_ptr[3 * i + 2]);
        out_degree[u]++;
        in_degree[v]++;
        adj_list[u].push_back({v, edge_id});
    }

    // --- FASE 1: MOTOR TOPOLÓGICO ---
    std::set<long long> anchor_nodes;
    for (size_t i = 0; i < num_nodes; ++i) {
        long long id = static_cast<long long>(nodes_ptr[3 * i]);
        if (in_degree[id] != 1 || out_degree[id] != 1)
            anchor_nodes.insert(id);
    }

    // Precompute angle threshold once to avoid sqrt/acos in the hot BFS loop
    const double cos_deviation_threshold = std::cos(angulo_maximo_desviacion * M_PI / 180.0);
    const double cos_threshold_sq        = cos_deviation_threshold * cos_deviation_threshold;

    std::set<long long> visited_edges;
    std::vector<std::vector<Point_pt>> all_original_paths;
    std::vector<std::pair<long long, long long>> final_edges_uv;

    // Collect new anchors separately to avoid mutating anchor_nodes while iterating it
    std::vector<long long> new_anchors;
    for (long long start_node : anchor_nodes) {
        for (const auto& edge : adj_list[start_node]) {
            long long next_node = edge.first;
            long long edge_id   = edge.second;

            if (visited_edges.count(edge_id)) continue;

            std::vector<long long> path_nodes = {start_node, next_node};
            visited_edges.insert(edge_id);
            long long current_node = next_node;

            while (anchor_nodes.find(current_node) == anchor_nodes.end()) {
                if (adj_list[current_node].empty()) break;

                auto     next_edge    = adj_list[current_node][0];
                long long next_v      = next_edge.first;
                long long next_edge_id = next_edge.second;

                Point_pt p1 = node_coords[path_nodes[path_nodes.size() - 2]];
                Point_pt p2 = node_coords[current_node];
                Point_pt p3 = node_coords[next_v];

                Vector_k v1(p1, p2);
                Vector_k v2(p2, p3);

                double sq_len1 = CGAL::to_double(v1.squared_length());
                double sq_len2 = CGAL::to_double(v2.squared_length());

                if (sq_len1 > 0.0 && sq_len2 > 0.0) {
                    double dot_product = CGAL::to_double(v1 * v2);
                    bool exceeds = (cos_deviation_threshold >= 0.0)
                        ? (dot_product < 0.0 || dot_product * dot_product < cos_threshold_sq * sq_len1 * sq_len2)
                        : (dot_product < 0.0 && dot_product * dot_product > cos_threshold_sq * sq_len1 * sq_len2);
                    if (exceeds) {
                        new_anchors.push_back(current_node);
                        break;
                    }
                }

                path_nodes.push_back(next_v);
                visited_edges.insert(next_edge_id);
                current_node = next_v;
            }

            std::vector<Point_pt> path_coords;
            path_coords.reserve(path_nodes.size());
            for (long long n : path_nodes) path_coords.push_back(node_coords[n]);

            if (path_coords.size() > 1) {
                all_original_paths.push_back(std::move(path_coords));
                final_edges_uv.push_back({start_node, current_node});
            }
        }
    }
    anchor_nodes.insert(new_anchors.begin(), new_anchors.end());

    // --- FASE 2: MOTOR GEOMÉTRICO (Voronoi Dual) ---
    // DT_Info stores node_id directly in each vertex, eliminating the float-key lookup map
    DT_Info dt;
    for (size_t i = 0; i < num_nodes; ++i) {
        long long id  = static_cast<long long>(nodes_ptr[3 * i]);
        auto      vh  = dt.insert(node_coords[id]);
        vh->info()    = id;
    }

    std::unordered_map<long long, double> node_epsilon;
    std::vector<double> valid_areas;

    for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
        long long node_id = vit->info();
        bool is_infinite  = false;
        std::vector<Point_pt> voronoi_vertices;

        auto fcirc  = dt.incident_faces(vit);
        auto fstart = fcirc;
        if (fcirc != nullptr) {
            do {
                if (dt.is_infinite(fcirc)) { is_infinite = true; break; }
                voronoi_vertices.push_back(dt.dual(fcirc));
                fcirc++;
            } while (fcirc != fstart);
        }

        if (!is_infinite && voronoi_vertices.size() >= 3) {
            double area = 0.0;
            size_t m    = voronoi_vertices.size();
            for (size_t j = 0; j < m; ++j) {
                size_t nj = (j + 1) % m;
                area += voronoi_vertices[j].x() * voronoi_vertices[nj].y()
                      - voronoi_vertices[nj].x() * voronoi_vertices[j].y();
            }
            area = 0.5 * std::abs(area);
            node_epsilon[node_id] = std::sqrt(area) * factor_epsilon;
            valid_areas.push_back(area);
        } else {
            node_epsilon[node_id] = -1.0;
        }
    }

    double global_avg_area       = valid_areas.empty() ? 100.0
        : std::accumulate(valid_areas.begin(), valid_areas.end(), 0.0) / valid_areas.size();
    double global_epsilon_fallback = std::sqrt(global_avg_area) * factor_epsilon;

    for (auto& pair : node_epsilon) {
        if (pair.second < 0.0) pair.second = global_epsilon_fallback;
    }

    // --- CONSTRUCCIÓN DEL AABB TREE ---
    std::vector<Segment_k> flatten_original_segments;
    for (const auto& path : all_original_paths) {
        for (size_t i = 0; i < path.size() - 1; ++i)
            flatten_original_segments.emplace_back(path[i], path[i + 1]);
    }

    AABB_Tree spatial_tree(flatten_original_segments.begin(), flatten_original_segments.end());
    if (with_index) spatial_tree.accelerate_distance_queries();

    // Compute all simplified paths first; enables exact pre-allocation of output buffers
    std::vector<std::vector<Point_pt>> all_simplified_paths(all_original_paths.size());
    // Hoist allocation outside nested loops; reused via clear()+reserve() each iteration
    std::vector<std::vector<Segment_k>::const_iterator> intersected_primitives;
    for (size_t i = 0; i < all_original_paths.size(); ++i) {
        long long u = final_edges_uv[i].first;
        long long v = final_edges_uv[i].second;

        double eps_u       = node_epsilon.count(u) ? node_epsilon[u] : global_epsilon_fallback;
        double eps_v       = node_epsilon.count(v) ? node_epsilon[v] : global_epsilon_fallback;
        double edge_epsilon = ((eps_u + eps_v) / 2.0) * factor_ajuste;

        std::vector<Point_pt> simplified_path = simplify_polyline(all_original_paths[i], edge_epsilon);

        if (with_index && simplified_path.size() > 2) {
            bool collides = false;
            for (size_t j = 0; j < simplified_path.size() - 1; ++j) {
                Segment_k test_seg(simplified_path[j], simplified_path[j + 1]);
                intersected_primitives.clear();
                intersected_primitives.reserve(4);
                spatial_tree.all_intersected_primitives(test_seg, std::back_inserter(intersected_primitives));
                if (intersected_primitives.size() > 2) { collides = true; break; }
            }
            if (collides)
                simplified_path = simplify_polyline(all_original_paths[i], 0.5);
        }
        all_simplified_paths[i] = std::move(simplified_path);
    }

    // Count exact output sizes for direct buffer pre-allocation (no intermediate vectors)
    size_t out_n_count = 0, out_e_count = 0;
    {
        std::set<long long> node_count_set;
        for (size_t i = 0; i < all_simplified_paths.size(); ++i) {
            const auto& sp = all_simplified_paths[i];
            if (sp.size() < 2) continue;
            long long u = final_edges_uv[i].first;
            long long v = final_edges_uv[i].second;
            out_e_count += sp.size() - 1;
            if (!node_count_set.count(u)) { ++out_n_count; node_count_set.insert(u); }
            if (!node_count_set.count(v)) { ++out_n_count; node_count_set.insert(v); }
        }
    }

    py::array_t<double> return_nodes({(py::ssize_t)out_n_count, (py::ssize_t)3});
    py::array_t<double> return_edges({(py::ssize_t)out_e_count, (py::ssize_t)3});
    double* nodes_out = return_nodes.mutable_data();
    double* edges_out = return_edges.mutable_data();
    std::set<long long> exported_nodes;
    size_t n_idx = 0, e_idx = 0;

    for (size_t i = 0; i < all_simplified_paths.size(); ++i) {
        const auto& simplified_path = all_simplified_paths[i];
        if (simplified_path.size() < 2) continue;
        long long u = final_edges_uv[i].first;
        long long v = final_edges_uv[i].second;

        if (!exported_nodes.count(u)) {
            nodes_out[n_idx * 3 + 0] = static_cast<double>(u);
            nodes_out[n_idx * 3 + 1] = simplified_path.front().x();
            nodes_out[n_idx * 3 + 2] = simplified_path.front().y();
            ++n_idx;
            exported_nodes.insert(u);
        }
        if (!exported_nodes.count(v)) {
            nodes_out[n_idx * 3 + 0] = static_cast<double>(v);
            nodes_out[n_idx * 3 + 1] = simplified_path.back().x();
            nodes_out[n_idx * 3 + 2] = simplified_path.back().y();
            ++n_idx;
            exported_nodes.insert(v);
        }

        for (size_t j = 0; j < simplified_path.size() - 1; ++j) {
            edges_out[e_idx * 3 + 0] = static_cast<double>(e_idx);
            edges_out[e_idx * 3 + 1] = static_cast<double>(u);
            edges_out[e_idx * 3 + 2] = static_cast<double>(v);
            ++e_idx;
        }
    }

    return SimplificationResult{py::make_tuple(return_nodes, return_edges)};
}
