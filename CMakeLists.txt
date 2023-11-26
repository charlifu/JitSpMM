cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(JitSpMM C CXX)

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

find_package(OpenMP REQUIRED)
find_package(MKL CONFIG REQUIRED)

find_package(asmjit REQUIRED
    PATHS ${CMAKE_SOURCE_DIR}/third_party/asmjit/build/lib64/cmake/asmjit)

set(CMAKE_BUILD_TYPE "Release")

include_directories(${CMAKE_SOURCE_DIR}/csrc/include)

add_executable(spmm csrc/spmm.cpp)
target_link_libraries(spmm PRIVATE OpenMP::OpenMP_CXX argparse asmjit::asmjit)
target_compile_features(spmm PRIVATE cxx_std_17)
target_compile_options(spmm PRIVATE -O3 -mavx512f)

add_executable(mm_to_csrbin csrc/mm_to_csrbin.cpp)
target_link_libraries(mm_to_csrbin argparse)
target_compile_features(mm_to_csrbin PRIVATE cxx_std_17)