# CMakeList.txt : CMake project for LinearAdvection
#

# cmake and global language config
cmake_minimum_required (VERSION 3.8)
project ("LinearAdvection" LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
enable_language(CUDA)

# set external libraries and locations
set(CUDA_LIBRARIES cudart_static cudadevrt cublas cublasLt curand cusolver cusparse)
# we're only using the linear algebra helper routines from opinf-cuda, 
# eventually this should be extracted and compiled as a separate library
set(OPINF_CUDA_DIR D:/Projects/operator-inference-cuda/operator-inference-cuda)
set(OPINF_CUDA_LIB ${OPINF_CUDA_DIR}/out/build/x64-Debug/opinf_cuda_lib.lib)

set(CUDPP_LIB_INCLUDE "D:/Projects/cudpp/include")
set(CUDPP_LIB_DIR "D:/Projects/cudpp/out/build/x64-Debug (default)/lib/cudpp64d.lib")

# collect all files in directory
file(GLOB LINEAR_ADVECTION_HEADERS include/*.h include/*.cuh)
file(GLOB LINEAR_ADVECTION_SRC src/*.cpp src/*.cu)

set(EXTERNAL_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OPINF_CUDA_DIR}/include ${CUDPP_LIB_INCLUDE})
set(EXTERNAL_LIBRARIES ${CUDA_LIBRARIES} ${OPINF_CUDA_LIB} ${CUDPP_LIB_DIR})

# first compile all to a library for modularity
add_library(linear_advection_lib STATIC ${LINEAR_ADVECTION_HEADERS} ${LINEAR_ADVECTION_SRC})
target_include_directories(linear_advection_lib PUBLIC ${EXTERNAL_INCLUDE_DIRS})
target_link_libraries(linear_advection_lib ${EXTERNAL_LIBRARIES})

# Build executable using project main file.
add_executable (LinearAdvection src/main.cu)
target_include_directories(LinearAdvection PUBLIC ${EXTERNAL_INCLUDE_DIRS})
target_link_libraries(LinearAdvection linear_advection_lib)

# TODO: Add tests and install targets if needed.
