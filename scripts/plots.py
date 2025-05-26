#!/usr/bin/env python3
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Valid OpenMP thread counts
VALID_THREADS = [2, 4, 8, 16, 32, 40]

def map_thread_count(n):
    """Round n up to the next valid thread count."""
    for t in VALID_THREADS:
        if n <= t:
            return t
    return VALID_THREADS[-1]

# Aggregation functions

def aggregate_serial(df):
    return df.groupby(['matrix', 'format'], as_index=False).agg({
        'duration_ms': 'median',
        'gflops': 'median',
        'rows': 'first',
        'cols': 'first',
        'nnz': 'first',
        'num_blocks': 'first'
    })


def aggregate_cuda(df):
    return df.groupby(['matrix', 'format', 'kernel', 'warps_per_block'], as_index=False).agg({
        'duration_ms': 'median',
        'gflops': 'median',
        'rows': 'first',
        'cols': 'first',
        'nnz': 'first',
        'num_blocks': 'first'
    })


def aggregate_openmp(df):
    df = df.copy()
    df['num_threads'] = df['num_threads'].apply(map_thread_count)
    return df.groupby(['matrix', 'format', 'bench', 'num_threads'], as_index=False).agg({
        'duration_ms': 'median',
        'gflops': 'median',
        'rows': 'first',
        'cols': 'first',
        'nnz': 'first',
        'num_blocks': 'first'
    })

# Plotting functions

def plot_serial(input_dir, output_dir):
    serial_csv = os.path.join(input_dir, 'serial.csv')
    if not os.path.isfile(serial_csv):
        raise FileNotFoundError(f"serial.csv not found in {input_dir}")
    df = pd.read_csv(serial_csv)
    df_agg = aggregate_serial(df)
    bench = 'spmv'

    for fmt in df_agg['format'].unique():
        df_fmt = df_agg[df_agg['format'] == fmt].set_index('matrix')
        out_dir = os.path.join(output_dir, 'serial', fmt, bench)
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        df_fmt['gflops'].sort_values().plot(kind='bar', width=0.6, ax=ax)
        ax.set_ylabel('GFLOPS')
        ax.set_xlabel('Matrix')
        ax.set_title(f'Serial GFLOPS per Matrix — {fmt}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = os.path.join(out_dir, f"{fmt}_gflops.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved plot: {out_file}")


def plot_cuda(input_dir, output_dir):
    cuda_csv = os.path.join(input_dir, 'cuda.csv')
    if not os.path.isfile(cuda_csv):
        raise FileNotFoundError(f"cuda.csv not found in {input_dir}")
    df = pd.read_csv(cuda_csv)
    df_agg = aggregate_cuda(df)
    bench = 'spmv'

    # Per-format per-wpb GFLOPS bars
    for fmt in df_agg['format'].unique():
        out_dir = os.path.join(output_dir, 'cuda', fmt, bench)
        os.makedirs(out_dir, exist_ok=True)
        for wpb in sorted(df_agg['warps_per_block'].unique()):
            df_sub = df_agg[(df_agg['format'] == fmt) & (df_agg['warps_per_block'] == wpb)]
            if df_sub.empty:
                continue
            pivot = df_sub.pivot(index='matrix', columns='kernel', values='gflops').fillna(0)
            pivot = pivot.loc[pivot.max(axis=1).sort_values().index]

            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind='bar', width=0.6, ax=ax)
            ax.set_ylabel('GFLOPS')
            ax.set_xlabel('Matrix')
            ax.set_title(f'CUDA GFLOPS per Matrix — {fmt}, wpb={wpb}')
            plt.xticks(rotation=45, ha='right')
            ax.legend(title='Kernel', bbox_to_anchor=(1, 1))
            plt.tight_layout()

            out_file = os.path.join(out_dir, f"cuda_{fmt.lower()}_gflops_wpb{wpb}.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved plot: {out_file}")

    # Best CSR vs HLL comparison across formats
    out_dir = os.path.join(output_dir, 'cuda', 'all', bench)
    os.makedirs(out_dir, exist_ok=True)
    best = df_agg.loc[df_agg.groupby(['matrix', 'format'])['gflops'].idxmax()]
    comp = best.pivot(index='matrix', columns='format', values='gflops')
    comp = comp.loc[comp.max(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, 6))
    comp.plot(kind='bar', width=0.8, ax=ax)
    ax.set_ylabel('GFLOPS')
    ax.set_xlabel('Matrix')
    ax.set_title('Best CUDA GFLOPS per Matrix: CSR vs HLL')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Format')
    plt.tight_layout()

    out_file = os.path.join(out_dir, 'cuda_best_csr_hll.png')
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved plot: {out_file}")


def plot_cuda_per_bin(input_dir, output_dir):
    cuda_csv = os.path.join(input_dir, 'cuda.csv')
    if not os.path.isfile(cuda_csv):
        raise FileNotFoundError(f"cuda.csv not found in {input_dir}")
    df = pd.read_csv(cuda_csv)

    # Considera solo warps_per_block 2, 4, 8
    df = df[df['warps_per_block'].isin([2, 4, 8])]

    # Binning per NNZ
    bins = [0, 10_000, 100_000, 500_000, 1_000_000, 2_500_000, 10_000_000, float('inf')]
    labels = ['<10K', '10K–100K', '100K–500K', '500K–1M', '1M–2.5M', '2.5M–10M', '≥10M']
    df['zero_bin'] = pd.cut(df['nnz'], bins=bins, labels=labels, right=False)

    bench = 'spmv'

    for kernel in df['kernel'].unique():
        df_kernel = df[df['kernel'] == kernel]

        for fmt in df_kernel['format'].unique():
            df_fmt = df_kernel[df_kernel['format'] == fmt]
            out_dir = os.path.join(output_dir, 'cuda', 'by_kernel', str(kernel), fmt, bench)
            os.makedirs(out_dir, exist_ok=True)

            grouped = df_fmt.groupby(['zero_bin', 'warps_per_block'])['gflops'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            for label in labels:
                df_zb = grouped[grouped['zero_bin'] == label]
                if df_zb.empty:
                    continue
                df_zb = df_zb.sort_values('warps_per_block')
                ax.plot(df_zb['warps_per_block'], df_zb['gflops'], marker='o', label=label)

            ax.set_xlabel('Warps per Block')
            ax.set_ylabel('GFLOPS')
            ax.set_title(f'CUDA Avg GFLOPS vs Warps per Block — Kernel: {kernel}, Format: {fmt}')
            ax.set_xticks([2, 4, 8])
            ax.legend(title='NNZ Bin', bbox_to_anchor=(1, 1))
            plt.tight_layout()

            out_file = os.path.join(out_dir, f"{str(kernel)}_{fmt}_avg_gflops_by_zeros.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved plot: {out_file}")



def plot_openmp(input_dir, output_dir):
    openmp_csv = os.path.join(input_dir, 'omp.csv')
    if not os.path.isfile(openmp_csv):
        raise FileNotFoundError(f"omp.csv not found in {input_dir}")
    df = pd.read_csv(openmp_csv)
    df_agg = aggregate_openmp(df)

    # Load serial for speedup
    serial_csv = os.path.join(input_dir, 'serial.csv')
    if not os.path.isfile(serial_csv):
        raise FileNotFoundError(f"serial.csv not found in {input_dir}")
    df_serial = pd.read_csv(serial_csv)
    df_ser = aggregate_serial(df_serial)[['matrix', 'format', 'duration_ms']]
    df_ser = df_ser.rename(columns={'duration_ms': 'serial_duration_ms'})
    df_agg = df_agg.merge(df_ser, on=['matrix', 'format'], how='left')
    df_agg['speedup'] = df_agg['serial_duration_ms'] / df_agg['duration_ms']

    # NNZ bins
    bins = [0, 10_000, 100_000, 500_000, 1_000_000, 2_500_000, 10_000_000, float('inf')]
    labels = ['<10K', '10K–100K', '100K–500K', '500K–1M', '1M–2.5M', '2.5M–10M', '≥10M']
    df_agg['zero_bin'] = pd.cut(df_agg['nnz'], bins=bins, labels=labels, right=False)

    for fmt in df_agg['format'].unique():
        for bench in df_agg['bench'].unique():
            subset = df_agg[(df_agg['format'] == fmt) & (df_agg['bench'] == bench)]
            if subset.empty:
                continue
            out_dir = os.path.join(output_dir, 'openmp', fmt, bench)
            os.makedirs(out_dir, exist_ok=True)

            # 1) GFLOPS per matrix (bar)
            pivot = subset.pivot(index='matrix', columns='num_threads', values='gflops').fillna(0)
            pivot = pivot[[t for t in VALID_THREADS if t in pivot.columns]]
            # sort matrices by increasing max GFLOPS across threads
            pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind='bar', width=0.8, ax=ax)
            ax.set_ylabel('GFLOPS')
            ax.set_xlabel('Matrix')
            ax.set_title(f'OpenMP GFLOPS per Matrix — {fmt}, {bench}')
            plt.xticks(rotation=45, ha='right')
            ax.legend(title='Threads', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            out_file = os.path.join(out_dir, f"{fmt}_{bench}_gflops.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved plot: {out_file}")

            # 2) Avg GFLOPS vs threads by zero-bin
            grouped = subset.groupby(['zero_bin', 'num_threads'])[['gflops', 'speedup']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            for zb in labels:
                df_z = grouped[grouped['zero_bin'] == zb]
                if df_z.empty:
                    continue
                df_z = df_z.sort_values('num_threads')
                ax.plot(df_z['num_threads'], df_z['gflops'], marker='o', label=zb)
            ax.set_xscale('log', base=2)
            ax.set_xticks(VALID_THREADS)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.set_xlabel('Threads')
            ax.set_ylabel('GFLOPS')
            ax.set_title(f'OpenMP Avg GFLOPS by NNZ Bin — {fmt}, {bench}')
            ax.legend(title='NNZ Bin', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            out_file = os.path.join(out_dir, f"{fmt}_{bench}_avg_gflops_by_zeros.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved plot: {out_file}")

            # 3) Avg Speedup vs threads by zero-bin
            fig, ax = plt.subplots(figsize=(10, 6))
            for zb in labels:
                df_z = grouped[grouped['zero_bin'] == zb]
                if df_z.empty:
                    continue
                df_z = df_z.sort_values('num_threads')
                ax.plot(df_z['num_threads'], df_z['speedup'], marker='o', label=zb)
            ax.set_xscale('log', base=2)
            ax.set_xticks(VALID_THREADS)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.set_xlabel('Threads')
            ax.set_ylabel('Speedup')
            ax.set_title(f'OpenMP Avg Speedup by NNZ Bin — {fmt}, {bench}')
            ax.legend(title='NNZ Bin', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            out_file = os.path.join(out_dir, f"{fmt}_{bench}_avg_speedup_by_zeros.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved plot: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GFLOPS bar charts for Serial, CUDA, and OpenMP SpMV benchmarks."
    )
    parser.add_argument('--bench-dir', required=True, help="Directory containing CSV benchmarks.")
    parser.add_argument('--out', default='plots', help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plot_serial(args.bench_dir, args.out)
    plot_openmp(args.bench_dir, args.out)
    plot_cuda(args.bench_dir, args.out)
    plot_cuda_per_bin(args.bench_dir, args.out)

if __name__ == '__main__':
    main()