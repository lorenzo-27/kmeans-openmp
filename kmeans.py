import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from kmeans_config import MAX_ITER, N_CLUSTERS, N_FEATURES_LIST, N_SAMPLES, NUM_THREADS

# Create necessary directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ImplementationType = Literal['AoS', 'SoA']

@dataclass
class ExperimentResult:
    threads: int
    sequential_time: float
    parallel_time: float
    speedup: float
    implementation: ImplementationType

def generate_dataset(
        n_samples: int, n_features: int, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset for clustering."""
    print(
        f"Generating dataset with {n_samples} samples, {n_features} features, and {n_clusters} clusters..."
    )
    x, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
    )
    return x, y

def save_dataset(x: np.ndarray, filename: str) -> None:
    """Save dataset to CSV file."""
    filepath = DATA_DIR / filename
    print(f"Saving dataset to '{filepath}'...")
    np.savetxt(filepath, x, delimiter=",")

def run_kmeans(
        input_file: str,
        k: int,
        max_iter: int,
        num_threads: List[int],
        implementation: ImplementationType
) -> pd.DataFrame:
    """Run k-means with different thread counts and collect results."""
    results = []
    input_path = str(DATA_DIR / input_file)
    impl_flag = "0" if implementation == "AoS" else "1"

    for threads in num_threads:
        print(f"Running k-means with {threads} threads using {implementation}...")
        cmd = [
            "./cmake-build-debug/kmeans",
            input_path,
            str(k),
            str(max_iter),
            str(threads),
            impl_flag
        ]
        print(f"Executing command: {' '.join(cmd)}")
        output = subprocess.check_output(cmd).decode()
        print(f"Output:\n{output}")

        # Parse timing results
        lines = output.strip().split("\n")
        seq_time = float(lines[1].split(": ")[1][:-2])
        par_time = float(lines[2].split(": ")[1][:-2])
        speedup = float(lines[3].split(": ")[1][:-1])

        results.append(
            ExperimentResult(
                threads=threads,
                sequential_time=seq_time,
                parallel_time=par_time,
                speedup=speedup,
                implementation=implementation
            )
        )

    return pd.DataFrame([vars(r) for r in results])

def plot_speedup_comparison(results_aos: pd.DataFrame, results_soa: pd.DataFrame, n_features: int) -> None:
    """Plot speedup comparison between AoS and SoA implementations."""
    output_path = RESULTS_DIR / f"speedup_comparison_{n_features}d.png"
    print(f"Plotting speedup comparison to '{output_path}'...")

    plt.figure(figsize=(12, 7))

    # Plot both implementations
    plt.plot(results_aos["threads"], results_aos["speedup"],
             marker="o", label="AoS", linewidth=2, markersize=8)
    plt.plot(results_soa["threads"], results_soa["speedup"],
             marker="s", label="SoA", linewidth=2, markersize=8)

    # Plot ideal speedup
    ideal_speedup = results_aos["threads"]
    plt.plot(results_aos["threads"], ideal_speedup, "--",
             label="Ideal speedup", alpha=0.5)

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"K-means Clustering Speedup Analysis ({n_features}D)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_execution_times_comparison(results_aos: pd.DataFrame, results_soa: pd.DataFrame, n_features: int) -> None:
    """Plot execution times comparison between AoS and SoA implementations."""
    output_path = RESULTS_DIR / f"execution_times_comparison_{n_features}d.png"
    print(f"Plotting execution times comparison to '{output_path}'...")

    plt.figure(figsize=(12, 7))

    # Plot sequential times
    plt.plot(results_aos["threads"], results_aos["sequential_time"],
             marker="s", label="AoS Sequential", linewidth=2, markersize=8)
    plt.plot(results_soa["threads"], results_soa["sequential_time"],
             marker="^", label="SoA Sequential", linewidth=2, markersize=8)

    # Plot parallel times
    plt.plot(results_aos["threads"], results_aos["parallel_time"],
             marker="o", label="AoS Parallel", linewidth=2, markersize=8)
    plt.plot(results_soa["threads"], results_soa["parallel_time"],
             marker="v", label="SoA Parallel", linewidth=2, markersize=8)

    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"K-means Clustering Execution Times ({n_features}D)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_kmeans_clustering(
        x: np.ndarray,
        labels: np.ndarray,
        n_features: int,
        n_clusters: int,
        implementation: ImplementationType,
) -> None:
    """Plot visual clustering."""
    output_path = RESULTS_DIR / f"kmeans_clustering_{implementation}_{n_features}d.png"
    print(f"Plotting K-means clustering ({implementation}) to '{output_path}'...")

    if n_features == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis")
        plt.title(f"K-means Clustering - {implementation} ({n_clusters} clusters)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig(output_path)
        plt.close()
    elif n_features == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels, cmap="viridis")
        ax.set_title(f"K-means Clustering - {implementation} ({n_clusters} clusters)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.savefig(output_path)
        plt.close()
    else:
        print(f"Plotting is only supported for 2D and 3D datasets. Current dimensions: {n_features}")

def main():
    """Run experiments for both AoS and SoA implementations"""
    implementations: List[ImplementationType] = ['AoS', 'SoA']

    for n_features in N_FEATURES_LIST:
        print(f"\nRunning experiments for {n_features} dimensions...")

        # Generate and save dataset
        x, y = generate_dataset(N_SAMPLES, n_features, N_CLUSTERS)
        dataset_file = f"dataset_{n_features}d.csv"
        save_dataset(x, dataset_file)

        # Store results for both implementations
        results_dict: Dict[ImplementationType, pd.DataFrame] = {}

        # Run experiments for both implementations
        for impl in implementations:
            print(f"\nRunning {impl} implementation...")
            results = run_kmeans(dataset_file, N_CLUSTERS, MAX_ITER, NUM_THREADS, impl)
            results_dict[impl] = results

            # Save individual results
            results_file = f"results_{impl}_{n_features}d.csv"
            results.to_csv(RESULTS_DIR / results_file, index=False)

            # Plot individual kmeans clustering
            plot_kmeans_clustering(x, y, n_features, N_CLUSTERS, impl)

        # Plot comparison results
        plot_speedup_comparison(results_dict['AoS'], results_dict['SoA'], n_features)
        plot_execution_times_comparison(results_dict['AoS'], results_dict['SoA'], n_features)

        # Print summary
        print("\nResults summary:")
        for impl in implementations:
            print(f"\n{impl} Implementation:")
            print(results_dict[impl])

if __name__ == "__main__":
    main()