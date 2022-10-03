cmake_minimum_required(VERSION 3.1)

project(beam_decode)


find_package(OpenCV REQUIRED)


include_directories(. ${OpenCV_INCLUDE_DIRS} $ENV{RAPIDJSON_DIR}/include/ $ENV{BOOST_DIR})


set(SRC_FILES beam_decode.cpp demo.cpp)
set(INC_FILES beam_decode.h)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${INC_FILES})

target_link_libraries(${PROJECT_NAME} 
${OpenCV_LIBS}
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_log-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_filesystem-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_date_time-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_thread-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_log_setup-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_chrono-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_atomic-vc142-mt-gd-x64-1_76.lib
)

