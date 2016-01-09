cmake_minimum_required(VERSION 2.8)
project(MandelbrotExplorer)

option(use_cuda "use_cuda" ON)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

if(use_cuda)
    find_package(CUDA QUIET REQUIRED)
    cuda_add_executable(MandelbrotExplorer src/main.cpp src/fractal.cu)
else()
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    add_executable(MandelbrotExplorer src/main.cpp src/fractal.cpp)
endif()

target_link_libraries(MandelbrotExplorer ${OpenCV_LIBS})

