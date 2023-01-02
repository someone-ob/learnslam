cmake_minimum_required(VERSION 3.0.0)
project(ch10 VERSION 0.1.0)

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

LIST(APPEND CMAKE_MODULE_PATH /home/jzh/g2o/cmake_modules)

Find_Package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)
include_directories("/usr/local/include/eigen3"
${Sophus_INCLUDE_DIRS}
${G2O_INCLUDE_DIRS})
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator
 g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse 
 g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff 
 g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
 
add_executable(pose_graph_g2o_SE3 src/pose_graph_g2o_SE3.cpp)

target_link_libraries(pose_graph_g2o_SE3
${G2O_LIBS})

add_executable(pose_graph_g2o_lie src/pose_graph_g2o_lie_algebra.cpp)

target_link_libraries(pose_graph_g2o_lie
${G2O_LIBS}
)

