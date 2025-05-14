cmake_minimum_required(VERSION 3.10)
project(spmv_scpa LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)

# Optimize aggressively; -fopenmp will be added automatically when we link OpenMP
set(CMAKE_C_FLAGS "-O3 -march=native")

add_compile_options(-ffast-math -funroll-loops)

include_directories(include)

# Find OpenMP
find_package(OpenMP REQUIRED)

set(SRC_FILES
    src/main.c
    src/mmio.c
    src/utils.c
    src/vector.c
    src/logger.c
    src/csr.c
    src/hll.c
)

# The main executable
add_executable(spmv_scpa ${SRC_FILES})

# Link everything together, including OpenMP
target_link_libraries(spmv_scpa
  # Link in OpenMP
  OpenMP::OpenMP_C
)
