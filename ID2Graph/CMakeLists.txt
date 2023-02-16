cmake_minimum_required(VERSION 3.13)
project("libllatvfl" LANGUAGES C CXX)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -march=native")

set(SOURCE_DIR  "src/llatvfl")
set(SCRIPT_DIR "script")
set(MPISCRIPT_DIR "mpiscript")
set(TEST_DIR  "test")

include(GNUInstallDirs)

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(${SOURCE_DIR})
add_subdirectory(${SCRIPT_DIR})
add_subdirectory(${MPISCRIPT_DIR})

enable_testing()
add_subdirectory(${TEST_DIR})