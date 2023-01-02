cmake_minimum_required(VERSION 3.0.0)
project(ch3 VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

#添加可执行文件
add_executable(eigenMatrix src/useEigen/eigenMatrix.cpp)
add_executable(eigenGeometry src/useGeometry/eigenGeometry.cpp)
add_executable(coordinateTransform src/examples/coordinateTransform.cpp)
add_executable(plotTrajectory src/examples/plotTrajectory.cpp)

#添加eigen头文件
find_package(Pangolin REQUIRED)
include_directories("/usr/local/include/eigen3"
${Pangolin_INCLUDE_DIRS}
"/home/jzh/Code/learnslam/ch3/include")
include(CPack)

add_library(draw lib/draw.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES}
draw)

