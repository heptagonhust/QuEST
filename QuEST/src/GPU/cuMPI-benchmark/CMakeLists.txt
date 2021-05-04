cmake_minimum_required(VERSION 2.8)

# Variables and Options
option(test "Build all tests." OFF) # use `cmake -Dtest=ON` to enable testing
set(CUDA_ARCH_CODE "sm_70")
set(CUDA_ARCH_COMPUTE "compute_70")
set (PROJECT_NAME cuMPI-benchmark)

project(${PROJECT_NAME})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add custom include/ directory
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
# include_directories(${COMMON_INCLUDES})

find_package(MPI REQUIRED)

find_package(NCCL REQUIRED)
if (NCCL_FOUND)
  add_definitions(-DUSE_NCCL)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
endif ()

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=${CUDA_ARCH_COMPUTE},code=${CUDA_ARCH_CODE};)

# Include directory
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  /usr/local/cuda/include
  ${MPI_INCLUDE_PATH}
)
# Link library directory
LINK_DIRECTORIES(
  /usr/lib
  /usr/local/lib
)

cuda_add_executable(
  ${PROJECT_NAME}
  main.cu
)
target_link_libraries(
  ${PROJECT_NAME}
  /lib64/libcuMPI.so
  ${NCCL_LIBRARY}
)