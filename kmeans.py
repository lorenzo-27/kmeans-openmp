import subprocess
from pathlib import Path
from typing import List, Tuple

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
    input_file: str, k: int, max_iter: int, num_threads: List[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run k-means with different thread counts and collect results."""
    results = []
    input_path = str(DATA_DIR / input_file)

    for threads in num_threads:
        print(f"Running k-means with {threads} threads...")
        # Run the C++ executable
        cmd = [
            "./cmake-build-debug/kmeans",
            input_path,
            str(k),
            str(max_iter),
            str(threads),
        ]
        print(f"Executing command: {' '.join(cmd)}")
        output = subprocess.check_output(cmd).decode()
        print(f"Output:\n{output}")

        # Parse timing results
        seq_time = float(output.split("\n")[0].split(": ")[1][:-2])
        par_time = float(output.split("\n")[1].split(": ")[1][:-2])
        speedup = float(output.split("\n")[2].split(": ")[1][:-1])

        results.append(
            {
                "threads": threads,
                "sequential_time": seq_time,
                "parallel_time": par_time,
                "speedup": speedup,
            }
        )

    return pd.DataFrame(results)


def plot_speedup(results: pd.DataFrame, output_file: str) -> None:
    """Plot speedup graph."""
    output_path = RESULTS_DIR / output_file
    print(f"Plotting speedup graph to '{output_path}'...")
    plt.figure(figsize=(10, 6))

    # Plot speedup
    plt.plot(
        results["threads"], results["speedup"], marker="o", linewidth=2, markersize=8
    )

    # Plot ideal speedup
    ideal_speedup = results["threads"]
    plt.plot(results["threads"], ideal_speedup, "--", label="Ideal speedup", alpha=0.5)

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title("K-means Clustering Speedup Analysis")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_execution_times(results: pd.DataFrame, output_file: str) -> None:
    """Plot execution times comparison."""
    output_path = RESULTS_DIR / output_file
    print(f"Plotting execution times comparison to '{output_path}'...")
    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threads"],
        results["sequential_time"],
        marker="s",
        label="Sequential",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        results["threads"],
        results["parallel_time"],
        marker="o",
        label="Parallel",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (ms)")
    plt.title("K-means Clustering Execution Times")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_kmeans_clustering(
    x: np.ndarray,
    labels: np.ndarray,
    n_features: int,
    n_clusters: int,
    output_file: str,
) -> None:
    """Plot visual clustering."""
    output_path = RESULTS_DIR / output_file
    print(f"Plotting K-means clustering to '{output_path}'...")

    if n_features == 2:
        # 2D plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis")
        plt.title(f"K-means Clustering ({n_clusters} clusters)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig(output_path)
        plt.close()
    elif n_features == 3:
        # 3D plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels, cmap="viridis")
        ax.set_title(f"K-means Clustering ({n_clusters} clusters)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.savefig(output_path)
        plt.close()
    else:
        print(
            f"Plotting is only supported for 2D and 3D datasets. Your dataset has {n_features} dimensions."
        )


def main():
    """Tests initialization"""
    # Run experiments for different dimensions
    for n_features in N_FEATURES_LIST:
        print(f"\nRunning experiments for {n_features} dimensions...")

        # Generate and save dataset
        x, y = generate_dataset(N_SAMPLES, n_features, N_CLUSTERS)
        dataset_file = f"dataset_{n_features}d.csv"
        save_dataset(x, dataset_file)

        # Run k-means and collect results
        print(f"Running k-means on dataset '{dataset_file}'...")
        results = run_kmeans(dataset_file, N_CLUSTERS, MAX_ITER, NUM_THREADS)

        # Save results to CSV
        results_file = f"results_{n_features}d.csv"
        results.to_csv(RESULTS_DIR / results_file, index=False)

        # Plot results
        plot_speedup(results, f"speedup_{n_features}d.png")
        plot_execution_times(results, f"execution_times_{n_features}d.png")

        # Plot k-means clustering
        plot_kmeans_clustering(
            x, y, n_features, N_CLUSTERS, f"kmeans_clustering_{n_features}d.png"
        )

        # Print summary
        print("\nResults summary:")
        print(results)


if __name__ == "__main__":
    main()

