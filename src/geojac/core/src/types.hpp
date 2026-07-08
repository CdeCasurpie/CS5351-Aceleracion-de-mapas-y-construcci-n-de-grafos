#pragma once

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
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>

typedef pybind11::tuple GraphData;

struct SimplificationResult {
    GraphData graph;
    std::unordered_map<int, std::vector<int>> node_lineage;
    std::unordered_map<int, std::vector<int>> edge_lineage;
};


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/create_straight_skeleton_2.h>
#include <CGAL/Straight_skeleton_2.h>

#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_2.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_2.h>
#include <CGAL/AABB_segment_primitive_2.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/intersections.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K>                   DT;
typedef CGAL::Triangulation_vertex_base_with_info_2<long long, K> Vb_info;
typedef CGAL::Triangulation_data_structure_2<Vb_info>             Tds_info;
typedef CGAL::Delaunay_triangulation_2<K, Tds_info>               DT_Info;
typedef K::Point_2                                          Point_pt;
typedef K::Segment_2                                        Segment_k;
typedef K::Vector_2                                         Vector_k;
typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_K;
typedef CGAL::Polygon_2<Exact_K>                          Polygon_exact;
typedef CGAL::Polygon_with_holes_2<Exact_K>               Polygon_with_holes_exact;
typedef CGAL::Polygon_set_2<Exact_K>                      Polygon_set_exact;
typedef CGAL::Straight_skeleton_2<Exact_K>                Straight_skeleton;
typedef CGAL::AABB_segment_primitive_2<K, std::vector<Segment_k>::const_iterator> Segment_Primitive;
typedef CGAL::AABB_traits_2<K, Segment_Primitive> AABB_Segment_Traits;
typedef CGAL::AABB_tree<AABB_Segment_Traits> AABB_Tree;
typedef std::shared_ptr<Straight_skeleton>                Straight_skeleton_ptr;
typedef std::tuple<long, long, long, Point_pt, Point_pt, Segment_k> SegmentInfo;



typedef CGAL::Polygon_2<K>                Polygon_fast;
typedef CGAL::Polygon_with_holes_2<K>     Polygon_with_holes_fast;
typedef CGAL::Straight_skeleton_2<K>      Straight_skeleton_fast;
typedef Straight_skeleton_fast::Vertex_const_handle Vertex_const_handle;

typedef CGAL::Search_traits_2<K> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree Tree;

namespace py = pybind11;

typedef std::unordered_map<long, Point_pt> NodeCoordMap;
typedef std::unordered_map<long, int> NodeDegreeMap;
typedef std::unordered_map<long, std::vector<long>> AdjacencyMap;
