# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

find_package(Threads REQUIRED)
# Find fmt
find_package(fmt CONFIG REQUIRED)

# Find gtest
find_package(GTest CONFIG REQUIRED)

message(STATUS "GTEST PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}")

#day01

file(GLOB_RECURSE DAY01_TEST_HEADER
        ${PROJECT_SOURCE_DIR}/test/day01/*.h
)
file(GLOB_RECURSE DAY01_TEST_SOURCES
        ${PROJECT_SOURCE_DIR}/test/day01/*.cc
)
add_executable(
        test_day01
        test_day01.cc
        ${DAY01_TEST_HEADER}
        ${DAY01_TEST_SOURCES}
)
target_link_libraries(
        test_day01
        PUBLIC
        cudaPoc
        PRIVATE
        fmt::fmt-header-only
        Threads::Threads
        GTest::gtest
        GTest::gtest_main
)
# Enable separable compilation
set_target_properties(test_day01 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(test_day01 PROPERTIES LINKER_LANGUAGE CUDA)

# Enable testing
enable_testing()

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(test_day01)