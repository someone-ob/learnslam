cmake_minimum_required(VERSION 3.0.0)
project(ch9 VERSION 0.1.0)

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
LIST(APPEND CMAKE_MODULE_PATH /home/jzh/g2o/cmake_modules )

Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)
find_package( G2O REQUIRED )
SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories("/home/jzh/Code/learnslam/ch9/include"
"/usr/local/include/eigen3"
${Sophus_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR} 
${CSPARSE_INCLUDE_DIR}
${G2O_INCLUDE_DIRS})

add_executable(bundle_adjustment_ceres src/bundle_adjustment_ceres.cpp)
add_executable(bundle_adjustment_g2o src/bundle_adjustment_g2o.cpp)
add_library(common "/home/jzh/Code/learnslam/ch9/lib/common.cpp")

target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES}
common)

target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS}
common)