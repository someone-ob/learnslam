cmake_minimum_required(VERSION 3.0.0)
project(ch4 VERSION 0.1.0)

enable_testing()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
include(CTest)
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS}
"/usr/local/include/eigen3")

add_executable(useSophus src/useSophus.cpp)

