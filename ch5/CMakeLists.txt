cmake_minimum_required(VERSION 3.0.0)
project(ch5 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(imageBasics src/imageBasics.cpp)
add_executable(undistortImage src/undistortImage.cpp)
add_executable(stereoVision src/stereoVision.cpp)
add_executable(joinMap src/joinMap.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# 链接OpenCV库
find_package( OpenCV REQUIRED )
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED)
#添加头文件
include_directories( ${OpenCV-INCLUDE_DIRS}
"/usr/local/include/eigen3"
${Pangolin_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS})

target_link_libraries(imageBasics ${OpenCV_LIBS})
target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(stereoVision ${OpenCV_LIBS}
${Pangolin_LIBRARIES})
target_link_libraries(joinMap ${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${Boost_LIBRARIES})
