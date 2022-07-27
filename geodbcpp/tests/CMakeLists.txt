
find_package(Torch REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/src)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


file(GLOB SRCS *.cpp)
add_executable(testgeodb ${SRCS})
target_link_libraries(testgeodb geodb GTest::gtest_main "${TORCH_LIBRARIES}")

include(GoogleTest)
gtest_discover_tests(testgeodb)
