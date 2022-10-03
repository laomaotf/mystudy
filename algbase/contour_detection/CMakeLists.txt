cmake_minimum_required(VERSION 3.1)

project(contour_det)

find_package(OpenCV REQUIRED)



aux_source_directory(source/ SRC_LIST)
include_directories(include/ ${OpenCV_INCLUDE_DIRS})

file(GLOB INC_LIST include/*.h)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
