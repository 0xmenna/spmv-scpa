import subprocess
import os
import argparse

def run_spmv(exec_path, matrix_base_path, result_dir, iterations):
    os.makedirs(result_dir, exist_ok=True)

    matrix_files = sorted([
        f for f in os.listdir(matrix_base_path)
        if f.endswith(".mtx") and os.path.isfile(os.path.join(matrix_base_path, f))
    ])

    if not matrix_files:
        print("[ERROR] No .mtx files found in the specified matrix directory.")
        return

    for matrix_file in matrix_files:
        matrix_path = os.path.join(matrix_base_path, matrix_file)
        matrix_name = os.path.splitext(matrix_file)[0]
        for i in range(iterations):
            print(f"[{matrix_name}] Iteration {i+1}/{iterations}")
            try:
                subprocess.run(
                    [exec_path, "-m", matrix_path, "-o", result_dir],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed for {matrix_name} at iteration {i+1}: {e}")

    print("All executions completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spmv executable over all .mtx files in a directory.")
    parser.add_argument("-exe", required=True, help="Path to the spmv executable")
    parser.add_argument("-m", required=True, help="Directory containing .mtx matrix files")
    parser.add_argument("-res", required=True, help="Directory to store result files")
    parser.add_argument("-i", type=int, default=10, help="Number of iterations per matrix")

    args = parser.parse_args()
    run_spmv(args.exe, args.m, args.res, args.i)
