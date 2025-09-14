cmake_minimum_required(VERSION 4.0)
project(hackmit LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

set(BUILD_GMOCK OFF CACHE BOOL "Disable building GoogleMock")
FetchContent_MakeAvailable(googletest)


enable_testing()
include_directories(${CURL_INCLUDE_DIR})

include(GoogleTest)



add_executable(hackmittest
        test.cpp
        test/tests.cpp
        main.cu
        test/timer.cpp
        src/sponge/sponge_gpu.cu
        src/sponge/sponge_cpu.cpp
        src/sponge/sponge.h
        src/sponge/file_manager.cpp
        src/sponge/spong_gpu_rel.cu

)
add_executable(hackmit main.cpp
        main.cu
        src/sponge/sponge_cpu.cpp
        src/sponge/sponge_gpu.cu
        src/sponge/sponge.h
        src/sponge/file_manager.cpp
        src/sponge/spong_gpu_rel.cu
)
# Link against GTest librariescak
target_link_libraries(hackmit
        PRIVATE
        CUDA::cublas
        CUDA::cusolver

)
target_link_libraries(hackmittest
        PRIVATE
        CUDA::cublas
        CUDA::cusolver
        gtest
        gtest_main

)
set_target_properties(hackmit PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_directories(hackmit PRIVATE ${CMAKE_SOURCE_DIR}/src/lib)

gtest_discover_tests(hackmittest)
