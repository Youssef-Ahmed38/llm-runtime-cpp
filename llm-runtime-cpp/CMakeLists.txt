cmake_minimum_required(VERSION 3.24)
project(aurora_runtime VERSION 0.1.0 LANGUAGES CXX)

option(AURORA_BUILD_TESTS "Build correctness tests" ON)
option(AURORA_BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(AURORA_ENABLE_OPENMP "Use OpenMP for CPU output-parallel kernels when available" ON)

add_library(aurora_runtime
    src/aurora_model_repository.cpp
    src/cpu_linear_algebra.cpp
    src/generation_service.cpp
    src/model.cpp
    src/runtime.cpp
    src/tensor.cpp
    src/tokenizer.cpp
)
target_include_directories(aurora_runtime PUBLIC include)
target_compile_features(aurora_runtime PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(aurora_runtime PRIVATE /W4 /permissive-)
else()
  target_compile_options(aurora_runtime PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(AURORA_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(aurora_runtime PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(aurora_runtime PRIVATE AURORA_HAS_OPENMP=1)
  endif()
endif()

add_executable(aurora app/main.cpp)
target_link_libraries(aurora PRIVATE aurora_runtime)

if(AURORA_BUILD_TESTS)
  enable_testing()
  add_executable(aurora_tests tests/test_runtime.cpp)
  target_link_libraries(aurora_tests PRIVATE aurora_runtime)
  add_test(NAME aurora_tests COMMAND aurora_tests)
endif()

if(AURORA_BUILD_BENCHMARKS)
  add_executable(aurora_benchmark benchmarks/benchmark_decode.cpp)
  target_link_libraries(aurora_benchmark PRIVATE aurora_runtime)
endif()
