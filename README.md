# Parallel Sparse Matrix-Vector Product (SpMV)

This project implements a **Sparse Matrix-Vector multiplication (SpMV)** benchmark in three variants:

- **Serial**: a baseline implementation, useful for validating results and comparison.
- **OpenMP**: a parallel CPU implementation using OpenMP to scale with multiple threads.
- **CUDA**: a GPU-accelerated implementation using different kernel strategies.

---

## 🔧 Build Instructions

To build the project:

```bash
mkdir build && cd build
cmake ..
make
```

## 🚀 Run Instructions

You can execute all benchmarks on a specific matrix file with:

```bash
./spmv -m <path_to_matrix_file.mtx> -o <path_to_outdir>
```

Arguments:
- -m: Path to the matrix file in Matrix Market format (.mtx)
- -o: Output directory where benchmark CSV results will be stored

This command will generate three CSV files in the output directory, containing benchmark results for the Serial, OpenMP, and CUDA variants.

## 📊 Automating Benchmarks Over Multiple Matrices

To benchmark all matrices across several iterations, use the provided Python script in the `scripts/` directory:

```bash
python3 results.py -exe <path_to_spmv_executable> -m <path_to_matrix_directory> -r <results_output_directory> -i <num_iterations_per_matrix>
```

Arguments:
- -exe: Path to the compiled spmv executable
- -m: Directory containing all matrix files (.mtx)
- -r: Directory where output result files will be written
- -i: Number of iterations to repeat each benchmark (for averaging)

This script will run the benchmark across all matrices in the directory and generate results suitable for performance analysis and plotting.
