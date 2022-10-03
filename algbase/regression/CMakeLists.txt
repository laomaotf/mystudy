cmake_minimum_required(VERSION 3.1)

project(regression)

find_package(OpenCV REQUIRED)


include_directories(include/ ${OpenCV_INCLUDE_DIRS} )

FILE(GLOB  SOURCE_FILES source/*.cpp)
FILE(GLOB HEAD_FILES include/*.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEAD_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
