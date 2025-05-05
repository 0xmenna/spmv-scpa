cmake_minimum_required(VERSION 3.5)
project(spmv_scpa C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O3 -march=native")

include_directories(include)


add_library(mmio_mod    src/mmio.c)
add_library(utils_mod   src/utils.c)
add_library(vec_mod     src/vector.c)
add_library(log_mod     src/logger.c)
add_library(csr_mod     src/csr.c)
add_library(hll_mod     src/hll.c)

# The main executable
add_executable(spmv_scpa src/main.c)

target_link_libraries(spmv_scpa
    PRIVATE
      log_mod       # main → logger
      csr_mod       # main → csr
      hll_mod       # csr → hll
      vec_mod       # csr/hll → vector
      utils_mod     # csr → utils
      mmio_mod      # csr → mmio
)
