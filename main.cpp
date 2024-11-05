#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <omp.h>

// Structure of Arrays (SoA) for better cache utilization and SIMD operations
struct Dataset {
    std::vector<float> coords; // Flattened coordinates [x1,y1,z1,...,x2,y2,z2,...]
    size_t n_points;
    size_t n_dims;
};

struct Centroids {
    std::vector<float> coords; // Flattened coordinates
    std::vector<size_t> counts; // Number of points in each cluster
    size_t k;
    size_t n_dims;
};

// Read dataset from CSV file
Dataset read_dataset(const std::string& filename) {
    Dataset data;
    std::ifstream file(filename);
    std::string line;

    // Read header to get dimensions
    std::getline(file, line);
    data.n_dims = std::count(line.begin(), line.end(), ',') + 1;

    // Read data
    std::vector<float> coords;
    while (std::getline(file, line)) {
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            coords.push_back(std::stof(line.substr(0, pos)));
            line.erase(0, pos + 1);
        }
        coords.push_back(std::stof(line));
    }

    data.n_points = coords.size() / data.n_dims;
    data.coords = std::move(coords);
    return data;
}

// Initialize centroids randomly (fixed seed for reproducibility)
Centroids initialize_centroids(const Dataset& data, size_t k) {
    std::mt19937 gen(42); // Fixed seed
    std::uniform_int_distribution<> dis(0, data.n_points - 1);

    Centroids centroids;
    centroids.k = k;
    centroids.n_dims = data.n_dims;
    centroids.coords.resize(k * data.n_dims);
    centroids.counts.resize(k);

    for (size_t i = 0; i < k; ++i) {
        size_t idx = dis(gen);
        for (size_t d = 0; d < data.n_dims; ++d) {
            centroids.coords[i * data.n_dims + d] =
                data.coords[idx * data.n_dims + d];
        }
    }
    return centroids;
}

// Compute squared Euclidean distance between a point and a centroid
float compute_distance(const float* point, const float* centroid, size_t n_dims) {
    float dist = 0.0f;
    #pragma omp simd reduction(+:dist)
    for (size_t d = 0; d < n_dims; ++d) {
        float diff = point[d] - centroid[d];
        dist += diff * diff;
    }
    return dist;
}

// Sequential k-means implementation
void kmeans_sequential(const Dataset& data, Centroids& centroids,
                      std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_coords(centroids.coords.size());
    std::vector<size_t> new_counts(centroids.k);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        for (size_t i = 0; i < data.n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max(); // Initialize to max value
            int best_cluster = 0; // Default to first cluster

            for (size_t j = 0; j < centroids.k; ++j) {
                float dist = compute_distance(
                    &data.coords[i * data.n_dims],
                    &centroids.coords[j * data.n_dims],
                    data.n_dims
                );
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update step
        std::fill(new_coords.begin(), new_coords.end(), 0.0f);
        std::fill(new_counts.begin(), new_counts.end(), 0);

        for (size_t i = 0; i < data.n_points; ++i) {
            int cluster = assignments[i];
            new_counts[cluster]++;

            for (size_t d = 0; d < data.n_dims; ++d) {
                new_coords[cluster * data.n_dims + d] +=
                    data.coords[i * data.n_dims + d];
            }
        }

        // Compute new centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (new_counts[j] > 0) {
                for (size_t d = 0; d < data.n_dims; ++d) {
                    centroids.coords[j * data.n_dims + d] =
                        new_coords[j * data.n_dims + d] / new_counts[j];
                }
            }
        }
    }
}

// Parallel k-means implementation using OpenMP
void kmeans_parallel(const Dataset& data, Centroids& centroids,
                    std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_coords(centroids.coords.size());
    std::vector<size_t> new_counts(centroids.k);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        #pragma omp parallel for
        for (size_t i = 0; i < data.n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max(); // Initialize to max value
            int best_cluster = 0; // Default to first cluster

            for (size_t j = 0; j < centroids.k; ++j) {
                float dist = compute_distance(
                    &data.coords[i * data.n_dims],
                    &centroids.coords[j * data.n_dims],
                    data.n_dims
                );
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update step
        std::fill(new_coords.begin(), new_coords.end(), 0.0f);
        std::fill(new_counts.begin(), new_counts.end(), 0);

        #pragma omp parallel default(none) shared(data, centroids, assignments, new_coords, new_counts)
        {
            // Local storage for each thread to avoid false sharing
            std::vector<float> local_coords(centroids.coords.size(), 0.0f);
            std::vector<size_t> local_counts(centroids.k, 0);

            #pragma omp for
            for (size_t i = 0; i < data.n_points; ++i) {
                int cluster = assignments[i];
                local_counts[cluster]++;

                for (size_t d = 0; d < data.n_dims; ++d) {
                    local_coords[cluster * data.n_dims + d] +=
                        data.coords[i * data.n_dims + d];
                }
            }

            #pragma omp critical
            {
                for (size_t j = 0; j < centroids.k; ++j) {
                    new_counts[j] += local_counts[j];
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        new_coords[j * data.n_dims + d] +=
                            local_coords[j * data.n_dims + d];
                    }
                }
            }
        }

        // Compute new centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (new_counts[j] > 0) {
                for (size_t d = 0; d < data.n_dims; ++d) {
                    centroids.coords[j * data.n_dims + d] =
                        new_coords[j * data.n_dims + d] / new_counts[j];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <k> <max_iter> <num_threads>\n";
        return 1;
    }

    std::string input_file = argv[1];
    int k = std::stoi(argv[2]);
    int max_iter = std::stoi(argv[3]);
    int num_threads = std::stoi(argv[4]);

    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Read dataset
    Dataset data = read_dataset(input_file);

    // Initialize centroids and assignments
    Centroids centroids = initialize_centroids(data, k);
    std::vector<int> assignments(data.n_points);

    // Run sequential version and measure time
    double seq_start = omp_get_wtime();
    kmeans_sequential(data, centroids, assignments, max_iter);
    double seq_end = omp_get_wtime();
    double seq_time = (seq_end - seq_start) * 1000; // Milliseconds

    // Run parallel version and measure time
    double par_start = omp_get_wtime();
    kmeans_parallel(data, centroids, assignments, max_iter);
    double par_end = omp_get_wtime();
    double par_time = (par_end - par_start) * 1000; // Milliseconds

    // Print timing results
    std::cout << "Sequential time: " << seq_time << "ms\n";
    std::cout << "Parallel time: " << par_time << "ms\n";
    std::cout << "Speedup: " << seq_time / par_time << "x\n";

    return 0;
}