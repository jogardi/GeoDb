cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -Wall -Wextra -g")
#execute_process(COMMAND $(python3 -c ") RESULT_VARIABLE FOO)
#message(STATUS "FOO: ${FOO}")
list(APPEND CMAKE_PREFIX_PATH /Users/joeyg/docs/workspace/libtorch)

set(CMAKE_CXX_STANDARD 20)

project(geodbcpp)

enable_testing()

include_directories(/usr/local/include)

find_package (glog 0.6.0 REQUIRED)
find_package(Torch REQUIRED)
#find_package(Python REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)
message(STATUS "CPLUS_INCLUDE_PATH: $ENV{CPLUS_INCLUDE_PATH}")

file(GLOB SOURCES "src/**")

add_library(geodb ${SOURCES})
target_link_libraries(geodb PRIVATE "${TORCH_LIBRARIES}" pybind11::embed glog::glog)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
pybind11_add_module(geodbcpp PyModule.cpp)
target_link_libraries(geodbcpp PRIVATE geodb)
set_property(TARGET geodbcpp PROPERTY CXX_STANDARD 20)


set(PYTHON_DIST_DIR /usr/local/Cellar/python@3.9/3.9.13_1/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages)
if (DEFINED $ENV{PYTHON_DIST_DIR})
    set(PYTHON_DIST_DIR $ENV{PYTHON_DIST_DIR})
endif ()

add_custom_command(TARGET geodbcpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:geodbcpp> ${PYTHON_DIST_DIR})


# Also copy our C++ shared lib to ${PYTHON_DIST_DIR}, which must be in unix env $LD_LIBRARY_PATH
add_custom_command(TARGET geodb POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:geodb> ${PYTHON_DIST_DIR})

add_subdirectory(tests)
