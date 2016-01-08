cmake_minimum_required(VERSION 2.8)
project(MandelbrotExplorer)
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
cuda_add_executable(MandelbrotExplorer src/main.cpp src/fractal.cu)
target_link_libraries(MandelbrotExplorer ${OpenCV_LIBS})

