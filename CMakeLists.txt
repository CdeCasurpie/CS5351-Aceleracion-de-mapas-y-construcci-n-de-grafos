cmake_minimum_required(VERSION 3.15)
project(acj_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Use FetchContent to extract the local CGAL tarball during build
include(FetchContent)
FetchContent_Declare(
  CGAL
  URL https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1.tar.xz
)
FetchContent_Populate(CGAL)

# PATCH CGAL BUG in AABB_traits_2.h
file(READ "${cgal_SOURCE_DIR}/include/CGAL/AABB_traits_2.h" AABB_TRAITS_CONTENT)
string(REPLACE "bb,true)?" "bb)?" AABB_TRAITS_CONTENT "${AABB_TRAITS_CONTENT}")
file(WRITE "${cgal_SOURCE_DIR}/include/CGAL/AABB_traits_2.h" "${AABB_TRAITS_CONTENT}")
# CGAL is header-only from 5.0+, just add the include directories
add_library(CGAL::CGAL INTERFACE IMPORTED)
target_include_directories(CGAL::CGAL INTERFACE "${cgal_SOURCE_DIR}/include")

# Find Boost (strictly required by CGAL)
if(NOT WIN32)
    find_package(Boost REQUIRED)
endif()

if(WIN32)
    # === WINDOWS LOGIC ===
    # Include vcpkg globally for gmp/mpfr and EXACTLY inject Boost include path like the experiment!
    include_directories("C:/vcpkg/installed/x64-windows/include")

    find_library(GMP_LIBRARY gmp PATHS "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
    find_library(MPFR_LIBRARY mpfr PATHS "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
    if(GMP_LIBRARY AND MPFR_LIBRARY)
        target_link_libraries(CGAL::CGAL INTERFACE ${GMP_LIBRARY} ${MPFR_LIBRARY})
    endif()

elseif(APPLE)
    # === MACOS LOGIC ===
    if(TARGET Boost::headers)
        target_link_libraries(CGAL::CGAL INTERFACE Boost::headers)
    elseif(TARGET Boost::boost)
        target_link_libraries(CGAL::CGAL INTERFACE Boost::boost)
    elseif(Boost_INCLUDE_DIRS)
        target_include_directories(CGAL::CGAL INTERFACE "${Boost_INCLUDE_DIRS}")
    endif()

    find_library(GMP_LIBRARY gmp)
    find_library(MPFR_LIBRARY mpfr)
    if(GMP_LIBRARY AND MPFR_LIBRARY)
        target_link_libraries(CGAL::CGAL INTERFACE ${GMP_LIBRARY} ${MPFR_LIBRARY})
    endif()

else()
    # === LINUX (UBUNTU) LOGIC ===
    if(TARGET Boost::headers)
        target_link_libraries(CGAL::CGAL INTERFACE Boost::headers)
    elseif(TARGET Boost::boost)
        target_link_libraries(CGAL::CGAL INTERFACE Boost::boost)
    elseif(Boost_INCLUDE_DIRS)
        target_include_directories(CGAL::CGAL INTERFACE "${Boost_INCLUDE_DIRS}")
    endif()

    find_library(GMP_LIBRARY gmp)
    find_library(MPFR_LIBRARY mpfr)
    if(GMP_LIBRARY AND MPFR_LIBRARY)
        target_link_libraries(CGAL::CGAL INTERFACE ${GMP_LIBRARY} ${MPFR_LIBRARY})
    endif()
endif()

add_compile_definitions(CGAL_DO_NOT_USE_BOOST_MP)
add_subdirectory(src/geojac/core)
