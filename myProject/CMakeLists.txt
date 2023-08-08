cmake_minimum_required(VERSION 3.1...3.14)
project(CgalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CGAL REQUIRED)

add_executable(CgalProject meshSimplification.cpp)
target_link_libraries(CgalProject CGAL::CGAL)

