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


#day05
#The .cu extension tells CMake to use nvcc instead of cl.exe
file(GLOB_RECURSE DAY05CU_TEST_HEADER
        ${PROJECT_SOURCE_DIR}/test/day05cu/*.h
)
file(GLOB_RECURSE DAY05CU_TEST_SOURCES
        ${PROJECT_SOURCE_DIR}/test/day05cu/*.cu
)
add_executable(
        test_day05cu
        test_day05cu.cc
        ${DAY05CU_TEST_HEADER}
        ${DAY05CU_TEST_SOURCES}
)
target_link_libraries(
        test_day05cu
        PUBLIC
        cudaPoc
        PRIVATE
        Threads::Threads
        GTest::gtest
        GTest::gtest_main
)
# Disable RDC for test files to avoid nvcc /utf-8 propagation issues
set_target_properties(test_day05cu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable testing
enable_testing()

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(test_day01)