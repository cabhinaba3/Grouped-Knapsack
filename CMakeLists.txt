cmake_minimum_required(VERSION 3.12)
project(GroupedKnapsack CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
endif()

# ParallelFor wraps tbb::parallel_for directly: libc++ does not implement the
# C++17 parallel algorithms, so std::execution::par is not an option here.
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Everything both binaries share: instance generation, the two algorithms,
# the experiment harness, the CLI parser and the parallel-for wrapper.
add_library(gka_core STATIC
    src/cli.cc
    src/dantzig.cc
    src/data_generate.cc
    src/distribution_flags.cc
    src/grouped.cc
    src/harness.cc
    src/parallel.cc
)
target_include_directories(gka_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gka_core PUBLIC TBB::tbb Threads::Threads Boost::program_options)

add_executable(gka_dantzig src/main.cc)
target_link_libraries(gka_dantzig PRIVATE gka_core)

add_subdirectory(matplotplusplus)

add_executable(gka_experiments src/experiments.cc)
target_link_libraries(gka_experiments PRIVATE gka_core Matplot++::matplot)
