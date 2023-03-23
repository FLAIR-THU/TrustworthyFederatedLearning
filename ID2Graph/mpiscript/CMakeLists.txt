cmake_minimum_required(VERSION 3.13)

set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpic++ )

add_executable(train_mpisecureboost pipeline_1_train_mpisecureboost.cpp)
# add_executable(train_mpirandomforest pipeline_1_train_mpirandomforest.cpp)
target_link_libraries(train_mpisecureboost boost_random boost_mpi boost_serialization pthread llatvfl)
# target_link_libraries(train_mpirandomforest boost_random boost_mpi boost_serialization pthread llatvfl)