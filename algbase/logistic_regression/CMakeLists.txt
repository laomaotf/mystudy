cmake_minimum_required(VERSION 2.6)

project(logreg)



find_package(OpenCV REQUIRED)



aux_source_directory(. SRC_LIST)
include_directories(. ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC_LIST} logreg.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
