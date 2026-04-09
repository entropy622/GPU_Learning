cmake_minimum_required(VERSION 3.20)
project(openmp_learning_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_library(openmp_lab_common INTERFACE)
target_include_directories(openmp_lab_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(openmp_lab_common INTERFACE OpenMP::OpenMP_CXX)

add_executable(omp_basics src/omp_basics.cpp)
target_link_libraries(omp_basics PRIVATE openmp_lab_common)

add_executable(omp_thread_sweep src/omp_thread_sweep.cpp)
target_link_libraries(omp_thread_sweep PRIVATE openmp_lab_common)

add_executable(omp_matrix_mul src/omp_matrix_mul.cpp)
target_link_libraries(omp_matrix_mul PRIVATE openmp_lab_common)

add_executable(omp_blocked_matmul_exercise src/omp_blocked_matmul_exercise.cpp)
target_link_libraries(omp_blocked_matmul_exercise PRIVATE openmp_lab_common)
