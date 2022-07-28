include_directories(${CMAKE_SOURCE_DIR}/src)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -Wall -Wextra -g")

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(pybind11 REQUIRED)

file(GLOB SRCS *.cpp)
add_executable(testgeodb ${SRCS})
target_link_libraries(testgeodb GTest::gtest_main "${TORCH_LIBRARIES}" geodb pybind11::embed)

include(GoogleTest)
gtest_discover_tests(testgeodb)
