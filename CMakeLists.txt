cmake_minimum_required(VERSION 2.8)
project(MandelbrotExplorer)
# comment out the next line if you don't have a CUDA supported device
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
# comment out the next line if you don't have a CUDA supported device
cuda_add_executable(MandelbrotExplorer src/main.cpp src/fractal.cu)
# uncomment the next line if you don't have a CUDA supported device
#add_executable(MandelbrotExplorer src/main.cpp src/fractal.cpp)
target_link_libraries(MandelbrotExplorer ${OpenCV_LIBS})

