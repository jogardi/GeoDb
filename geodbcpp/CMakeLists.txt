cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

project(geodbcpp)

enable_testing()

include_directories(/usr/local/include)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

file(GLOB SOURCES "src/**")

add_library(geodb ${SOURCES})
target_link_libraries(geodb PUBLIC "${TORCH_LIBRARIES}")
#pybind11_add_module(geodbpy PyModule.cpp)
#target_link_libraries(geodbpy PRIVATE geodb)
#
##install(TARGETS geodb COMPONENT python LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}")
#
#
#set(PYTHON_DIST_DIR /usr/local/lib/python3.8/site-packages)
#if (DEFINED $ENV{PYTHON_DIST_DIR})
#    set(PYTHON_DIST_DIR $ENV{PYTHON_DIST_DIR})
#endif ()
#
#add_custom_command(TARGET geodbpy POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:geodbpy> ${PYTHON_DIST_DIR})
#
#
## Also copy our C++ shared lib to ${PYTHON_DIST_DIR}, which must be in unix env $LD_LIBRARY_PATH
#add_custom_command(TARGET geodb POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:geodb> ${PYTHON_DIST_DIR})

add_subdirectory(tests)
