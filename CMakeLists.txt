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

# libstdc++ and libc++ implement the C++17 parallel algorithms on top of TBB,
# so std::execution::par needs it linked, not just the <execution> header.
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

# Everything both binaries share: instance generation, the two algorithms,
# the experiment harness, the CLI parser and the parallel-for wrapper.
add_library(gka_core STATIC
    src/dataGenerate.cpp
    src/dantzig.cpp
    src/grouped.cpp
    src/harness.cpp
    src/parallel.cpp
    src/cli.cpp
)
target_include_directories(gka_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gka_core PUBLIC TBB::tbb Threads::Threads)

add_executable(gka_dantzig src/main.cpp)
target_link_libraries(gka_dantzig PRIVATE gka_core)

find_package(Matplot++ REQUIRED)


add_executable(gka_experiments src/experiments.cpp)
target_link_libraries(gka_experiments PRIVATE gka_core Matplot++::matplot)
