cmake_minimum_required(VERSION 3.0.0)
project(ch7 VERSION 0.1.0)

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")


add_executable(orb_cv src/orb_cv.cpp)
add_executable(orb_self src/orb_self.cpp)
add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)
add_executable(triangulation src/triangulation.cpp)
add_executable(pose_estimation_3d2d src/pose_estimation_3d2d.cpp)
add_executable(pose_estimation_3d3d src/pose_estimation_3d3d.cpp)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
LIST( APPEND CMAKE_MODULE_PATH /home/jzh/g2o/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
 
include_directories(
${OpenCV_INCLUDE_DIRS}
${G2O_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
"/usr/local/include/eigen3"
"/home/jzh/Code/learnslam/ch7/include")

add_library(FindFeatureMatches /home/jzh/Code/learnslam/ch7/lib/FindFeatureMatches.cpp)
add_library(poseEstimation2d2d /home/jzh/Code/learnslam/ch7/lib/poseEstimation2d2d.cpp)
add_library(pixel2cam /home/jzh/Code/learnslam/ch7/lib/pixel2cam.cpp)
add_library(bundleAdjustmentGaussNewton /home/jzh/Code/learnslam/ch7/lib/bundleAdjustmentGaussNewton.cpp)

target_link_libraries(orb_cv ${OpenCV_LIBS})
target_link_libraries(orb_self ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS}
FindFeatureMatches
poseEstimation2d2d
pixel2cam)
target_link_libraries(triangulation ${OpenCV_LIBS}
FindFeatureMatches
poseEstimation2d2d
pixel2cam)

target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} fmt 
${G2O_LIBS}
FindFeatureMatches
poseEstimation2d2d
pixel2cam
bundleAdjustmentGaussNewton)

target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} fmt 
${G2O_LIBS}
FindFeatureMatches
poseEstimation2d2d
pixel2cam)