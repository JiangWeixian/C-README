cmake_minimum_required(VERSION 3.3)
project(readme)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
include(ExternalProject)
find_package (Glog REQUIRED)
find_package (OpenCV REQUIRED)
include_directories (BEFORE ${GLOG_INCLUDE_DIR} ${OpenCV_LIBS})

set(SOURCE_FILES main.cpp str/segmentation.h video/camera.h pic/measure.h util/arr.h)
add_executable(readme ${SOURCE_FILES})
target_link_libraries (readme glog ${OpenCV_LIBS})
