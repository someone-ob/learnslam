cmake_minimum_required(VERSION 3.0.0)
project(ch8 VERSION 0.1.0)

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

add_executable(optical_flow src/optical_flow.cpp)
add_executable(direct_method src/direct_method.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
include_directories("/usr/local/include/eigen3"
${Pangolin_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS})

target_link_libraries(optical_flow ${OpenCV_LIBS} fmt)
target_link_libraries(direct_method ${OpenCV_LIBS} fmt
${Pangolin_LIBRARIES})