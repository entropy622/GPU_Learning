cmake_minimum_required(VERSION 3.20)
project(modern_cpp_perf_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MCPLAB_ENABLE_MPI "Build MPI examples when MPI is available" ON)

add_library(mcplab_common INTERFACE)
target_include_directories(mcplab_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(raii_demo src/raii_demo.cpp)
target_link_libraries(raii_demo PRIVATE mcplab_common)

add_executable(move_semantics_demo src/move_semantics_demo.cpp)
target_link_libraries(move_semantics_demo PRIVATE mcplab_common)

add_executable(smart_pointers_demo src/smart_pointers_demo.cpp)
target_link_libraries(smart_pointers_demo PRIVATE mcplab_common)

add_executable(concurrency_counter_demo src/concurrency_counter_demo.cpp)
target_link_libraries(concurrency_counter_demo PRIVATE mcplab_common)

add_executable(memory_order_demo src/memory_order_demo.cpp)
target_link_libraries(memory_order_demo PRIVATE mcplab_common)

if(MCPLAB_ENABLE_MPI)
  find_package(MPI QUIET COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    add_executable(mpi_basics_demo src/mpi_basics_demo.cpp)
    target_link_libraries(mpi_basics_demo PRIVATE mcplab_common MPI::MPI_CXX)
    target_compile_definitions(mpi_basics_demo PRIVATE MCPLAB_HAS_MPI=1)
  else()
    message(STATUS "MPI not found. mpi_basics_demo will not be built.")
  endif()
endif()
