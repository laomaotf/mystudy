cmake_minimum_required(VERSION 3.1)

project(watermark)

find_package(OpenCV REQUIRED)



aux_source_directory(. SRC_LIST)
include_directories(. ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC_LIST} watermark.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
