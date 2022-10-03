cmake_minimum_required(VERSION 3.1)

project(inpaint_image)

find_package(OpenCV REQUIRED)


file(GLOB_RECURSE INC_LIST  inc/*.h inc/*.hpp) 
source_group("inc" FILES ${INC_LIST})


aux_source_directory(src SRC_LIST)
include_directories(. ${OpenCV_INCLUDE_DIRS} inc/)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
