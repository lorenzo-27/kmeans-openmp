#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <omp.h>

// Tag dispatch for implementation selection
struct SoATag {};
struct AoSTag {};

// Base Dataset structure with common functionality
struct DatasetBase {
    size_t n_points;
    size_t n_dims;
};

// Structure of Arrays (SoA) implementation
struct DatasetSoA : DatasetBase {
    std::vector<std::vector<float>> coords_by_dim; // [dim][point]

    void resize(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        coords_by_dim.resize(dims);
        for (auto& dim : coords_by_dim) {
            dim.resize(points);
        }
    }
};

// Array of Structures (AoS) implementation
struct DatasetAoS : DatasetBase {
    std::vector<float> coords; // [point * n_dims + dim]

    void resize(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        coords.resize(points * dims);
    }
};

// Base Centroids structure
struct CentroidsBase {
    size_t k;
    size_t n_dims;
    std::vector<size_t> counts;
};

// SoA implementation for centroids
struct CentroidsSoA : CentroidsBase {
    std::vector<std::vector<float>> coords_by_dim; // [dim][centroid]

    void resize(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        coords_by_dim.resize(dims);
        for (auto& dim : coords_by_dim) {
            dim.resize(clusters);
        }
        counts.resize(clusters);
    }
};

// AoS implementation for centroids
struct CentroidsAoS : CentroidsBase {
    std::vector<float> coords; // [centroid * n_dims + dim]

    void resize(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        coords.resize(clusters * dims);
        counts.resize(clusters);
    }
};

template<typename Dataset>
Dataset read_dataset(const std::string& filename) {
    Dataset data;
    std::ifstream file(filename);
    std::string line;

    // Read header to get dimensions
    std::getline(file, line);
    size_t n_dims = std::count(line.begin(), line.end(), ',') + 1;

    // Count points
    size_t n_points = 1; // Including the first line we just read
    while (std::getline(file, line)) {
        n_points++;
    }

    // Resize the dataset
    data.resize(n_points, n_dims);

    // Reset file pointer and skip header
    file.clear();
    file.seekg(0);
    std::getline(file, line);

    if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
        // SoA reading implementation
        size_t point_idx = 0;
        while (std::getline(file, line)) {
            size_t pos = 0;
            size_t dim_idx = 0;
            while ((pos = line.find(',')) != std::string::npos) {
                data.coords_by_dim[dim_idx][point_idx] = std::stof(line.substr(0, pos));
                line.erase(0, pos + 1);
                dim_idx++;
            }
            data.coords_by_dim[dim_idx][point_idx] = std::stof(line);
            point_idx++;
        }
    } else {
        // AoS reading implementation
        size_t point_idx = 0;
        while (std::getline(file, line)) {
            size_t pos = 0;
            size_t dim_idx = 0;
            while ((pos = line.find(',')) != std::string::npos) {
                data.coords[point_idx * data.n_dims + dim_idx] = std::stof(line.substr(0, pos));
                line.erase(0, pos + 1);
                dim_idx++;
            }
            data.coords[point_idx * data.n_dims + dim_idx] = std::stof(line);
            point_idx++;
        }
    }

    return data;
}

template<typename Dataset, typename Centroids>
Centroids initialize_centroids(const Dataset& data, size_t k) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, data.n_points - 1);

    Centroids centroids;
    centroids.resize(k, data.n_dims);

    if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
        // SoA implementation
        for (size_t i = 0; i < k; ++i) {
            size_t idx = dis(gen);
            for (size_t d = 0; d < data.n_dims; ++d) {
                centroids.coords_by_dim[d][i] = data.coords_by_dim[d][idx];
            }
        }
    } else {
        // AoS implementation
        for (size_t i = 0; i < k; ++i) {
            size_t idx = dis(gen);
            for (size_t d = 0; d < data.n_dims; ++d) {
                centroids.coords[i * data.n_dims + d] =
                    data.coords[idx * data.n_dims + d];
            }
        }
    }

    return centroids;
}

template<typename Dataset, typename Centroids>
float compute_distance(const Dataset& data, size_t point_idx,
                      const Centroids& centroids, size_t centroid_idx) {
    float dist = 0.0f;

    if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
        // SoA implementation
        #pragma omp simd reduction(+:dist)
        for (size_t d = 0; d < data.n_dims; ++d) {
            float diff = data.coords_by_dim[d][point_idx] -
                        centroids.coords_by_dim[d][centroid_idx];
            dist += diff * diff;
        }
    } else {
        // AoS implementation
        #pragma omp simd reduction(+:dist)
        for (size_t d = 0; d < data.n_dims; ++d) {
            float diff = data.coords[point_idx * data.n_dims + d] -
                        centroids.coords[centroid_idx * data.n_dims + d];
            dist += diff * diff;
        }
    }

    return dist;
}

template<typename Dataset, typename Centroids>
void kmeans_sequential(const Dataset& data, Centroids& centroids,
                      std::vector<int>& assignments, int max_iter) {
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        for (size_t i = 0; i < data.n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (size_t j = 0; j < centroids.k; ++j) {
                float dist = compute_distance(data, i, centroids, j);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Reset centroids
        if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
            for (auto& dim : centroids.coords_by_dim) {
                std::fill(dim.begin(), dim.end(), 0.0f);
            }
        } else {
            std::fill(centroids.coords.begin(), centroids.coords.end(), 0.0f);
        }
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0);

        // Update step
        for (size_t i = 0; i < data.n_points; ++i) {
            int cluster = assignments[i];
            centroids.counts[cluster]++;

            if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
                for (size_t d = 0; d < data.n_dims; ++d) {
                    centroids.coords_by_dim[d][cluster] += data.coords_by_dim[d][i];
                }
            } else {
                for (size_t d = 0; d < data.n_dims; ++d) {
                    centroids.coords[cluster * data.n_dims + d] +=
                        data.coords[i * data.n_dims + d];
                }
            }
        }

        // Normalize centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (centroids.counts[j] > 0) {
                if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.coords_by_dim[d][j] /= centroids.counts[j];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.coords[j * data.n_dims + d] /= centroids.counts[j];
                    }
                }
            }
        }
    }
}

template<typename Dataset, typename Centroids>
void kmeans_parallel(const Dataset& data, Centroids& centroids,
                    std::vector<int>& assignments, int max_iter) {
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        #pragma omp parallel for
        for (size_t i = 0; i < data.n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (size_t j = 0; j < centroids.k; ++j) {
                float dist = compute_distance(data, i, centroids, j);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Reset centroids
        if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
            for (auto& dim : centroids.coords_by_dim) {
                std::fill(dim.begin(), dim.end(), 0.0f);
            }
        } else {
            std::fill(centroids.coords.begin(), centroids.coords.end(), 0.0f);
        }
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0);

        // Update step with thread-local storage
        #pragma omp parallel default(none) shared(data, centroids, assignments)
        {
            Centroids local_centroids;
            local_centroids.resize(centroids.k, data.n_dims);
            std::vector<size_t> local_counts(centroids.k, 0);

            #pragma omp for
            for (size_t i = 0; i < data.n_points; ++i) {
                int cluster = assignments[i];
                local_counts[cluster]++;

                if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        local_centroids.coords_by_dim[d][cluster] += data.coords_by_dim[d][i];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        local_centroids.coords[cluster * data.n_dims + d] +=
                            data.coords[i * data.n_dims + d];
                    }
                }
            }

            // Merge thread-local results using atomic operations
            for (size_t j = 0; j < centroids.k; ++j) {
                #pragma omp atomic
                centroids.counts[j] += local_counts[j];

                if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        #pragma omp atomic
                        centroids.coords_by_dim[d][j] += local_centroids.coords_by_dim[d][j];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        #pragma omp atomic
                        centroids.coords[j * data.n_dims + d] +=
                            local_centroids.coords[j * data.n_dims + d];
                    }
                }
            }
        }

        // Normalize centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (centroids.counts[j] > 0) {
                if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.coords_by_dim[d][j] /= centroids.counts[j];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.coords[j * data.n_dims + d] /= centroids.counts[j];
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <k> <max_iter> <num_threads> <implementation>\n"
                  << "implementation: 0 for AoS, 1 for SoA\n";
        return 1;
    }

    std::string input_file = argv[1];
    int k = std::stoi(argv[2]);
    int max_iter = std::stoi(argv[3]);
    int num_threads = std::stoi(argv[4]);
    int impl = std::stoi(argv[5]);

    omp_set_num_threads(num_threads);

    if (impl == 0) {
        // AoS implementation
        auto data = read_dataset<DatasetAoS>(input_file);
        auto centroids = initialize_centroids<DatasetAoS, CentroidsAoS>(data, k);
        std::vector<int> assignments(data.n_points);

        double seq_start = omp_get_wtime();
        kmeans_sequential<DatasetAoS, CentroidsAoS>(data, centroids, assignments, max_iter);
        double seq_end = omp_get_wtime();
        double seq_time = (seq_end - seq_start) * 1000;

        double par_start = omp_get_wtime();
        kmeans_parallel<DatasetAoS, CentroidsAoS>(data, centroids, assignments, max_iter);
        double par_end = omp_get_wtime();
        double par_time = (par_end - par_start) * 1000;

        std::cout << "Implementation: AoS\n";
        std::cout << "Sequential time: " << seq_time << "ms\n";
        std::cout << "Parallel time: " << par_time << "ms\n";
        std::cout << "Speedup: " << seq_time / par_time << "x\n";
    } else {
        // SoA implementation
        auto data = read_dataset<DatasetSoA>(input_file);
        auto centroids = initialize_centroids<DatasetSoA, CentroidsSoA>(data, k);
        std::vector<int> assignments(data.n_points);

        double seq_start = omp_get_wtime();
        kmeans_sequential<DatasetSoA, CentroidsSoA>(data, centroids, assignments, max_iter);
        double seq_end = omp_get_wtime();
        double seq_time = (seq_end - seq_start) * 1000;

        double par_start = omp_get_wtime();
        kmeans_parallel<DatasetSoA, CentroidsSoA>(data, centroids, assignments, max_iter);
        double par_end = omp_get_wtime();
        double par_time = (par_end - par_start) * 1000;

        std::cout << "Implementation: SoA\n";
        std::cout << "Sequential time: " << seq_time << "ms\n";
        std::cout << "Parallel time: " << par_time << "ms\n";
        std::cout << "Speedup: " << seq_time / par_time << "x\n";
    }

    return 0;
}