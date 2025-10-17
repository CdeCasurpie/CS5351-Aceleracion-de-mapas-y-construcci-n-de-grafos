cmake_minimum_required(VERSION 3.15)
project(matcher_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Release mode for optimizations if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find CGAL
find_package(CGAL REQUIRED)

# Create the Python module
pybind11_add_module(matcher src/match.cpp)

# Link CGAL libraries
target_link_libraries(matcher PRIVATE CGAL::CGAL)
