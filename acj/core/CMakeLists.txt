cmake_minimum_required(VERSION 3.15)
project(acj_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Release mode for optimizations if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CGAL REQUIRED)

# Create the Python module
pybind11_add_module(acj_core src/acj_core.cpp)

# Link CGAL libraries
target_link_libraries(acj_core PRIVATE CGAL::CGAL)

# Set output directory to project build folder
set_target_properties(acj_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
