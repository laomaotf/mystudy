cmake_minimum_required(VERSION 3.1)

project(dwt)

find_package(OpenCV REQUIRED)



aux_source_directory(source/ SRC_LIST)
include_directories(. ${OpenCV_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp dwt.h dwt.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
