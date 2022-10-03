cmake_minimum_required(VERSION 3.1)

project(testbed)

include_directories(inc/ $ENV{BOOST_DIR})

set(SRC src/demo.cpp src/basic_tree.cpp src/utils.cpp)
set(INC inc/basic_tree.h inc/utils.h)

add_executable(${PROJECT_NAME} ${SRC} ${INC})

target_link_libraries(${PROJECT_NAME} 
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_log-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_filesystem-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_date_time-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_thread-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_log_setup-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_chrono-vc142-mt-gd-x64-1_76.lib
$ENV{BOOST_DIR}/lib64-msvc-14.2/libboost_atomic-vc142-mt-gd-x64-1_76.lib
)


