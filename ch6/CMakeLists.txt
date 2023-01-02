cmake_minimum_required(VERSION 3.0.0)
project(ch6 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(gaussNewton src/gaussNewton.cpp)
add_executable(ceresCurveFitting src/ceresCurveFitting.cpp)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories("/usr/include/eigen3"
${OpenCV_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS})

target_link_libraries(gaussNewton ${OpenCV_LIBS})
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS}
${CERES_LIBRARIES})