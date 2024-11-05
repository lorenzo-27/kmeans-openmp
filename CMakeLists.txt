cmake_minimum_required(VERSION 3.29)
project(kmeans)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(kmeans ${SOURCE_FILES})
