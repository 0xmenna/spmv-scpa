cmake_minimum_required(VERSION 3.5)
project(spmv_scpa C)

set(CMAKE_C_STANDARD 99)

# Optimize aggressively; -fopenmp will be added automatically when we link OpenMP
set(CMAKE_C_FLAGS "-O3 -march=native")

include_directories(include)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Libraries
add_library(mmio_mod    src/mmio.c)
add_library(utils_mod   src/utils.c)
add_library(vec_mod     src/vector.c)
add_library(log_mod     src/logger.c)
add_library(csr_mod     src/csr.c)
add_library(hll_mod     src/hll.c)

# The main executable
add_executable(spmv_scpa src/main.c)

# Link everything together, including OpenMP
target_link_libraries(spmv_scpa
    PRIVATE
      log_mod
      csr_mod
      hll_mod
      vec_mod
      utils_mod
      mmio_mod

      # Link in OpenMP
      OpenMP::OpenMP_C
)

# (Optional) If you want to force a minimum OpenMP version:
# target_link_libraries(spmv_scpa PRIVATE OpenMP::OpenMP_C@4.5)
#
# And if you use CXX for C++, you’d do OpenMP::OpenMP_CXX
