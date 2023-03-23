cmake_minimum_required(VERSION 3.13)

set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpic++ )

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.0.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(TEST_NAME llatvfl-test-mpi)

file(GLOB testSRC
    "*.cpp"
)
set(TEST_RUNNER main.cpp ${testSRC})
add_executable(${TEST_NAME} ${TEST_RUNNER})

target_link_libraries(${TEST_NAME} gtest gtest_main boost_random boost_mpi boost_serialization pthread llatvfl)

set_property(TARGET ${TEST_NAME} PROPERTY CROSSCOMPILING_EMULATOR 'mpirun --allow-run-as-root -np 2')

gtest_discover_tests(${TEST_NAME})