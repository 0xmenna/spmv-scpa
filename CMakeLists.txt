cmake_minimum_required(VERSION 3.10)
project(spmv_scpa LANGUAGES C CXX CUDA)


set(CMAKE_C_STANDARD 99)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Iinclude -O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -lm")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options=-fopenmp")

add_compile_options(-ffast-math -funroll-loops)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

include_directories(include)

set(SRC_FILES
    src/main.c
    src/mmio.c
    src/utils.c
    src/vector.c
    src/logger.c
    src/csr.c
    src/hll.c
    src/cuda_csr.cu
    src/cuda_hll.cu
    src/cuda_timer.cu 
)

# Crea l'eseguibile
add_executable(spmv_scpa ${SRC_FILES})

# Linka OpenMP
target_link_libraries(spmv_scpa OpenMP::OpenMP_CXX)

# Abilita la compilazione separabile per CUDA
set_target_properties(spmv_scpa PROPERTIES CUDA_SEPARABLE_COMPILATION ON)