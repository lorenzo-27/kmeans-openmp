#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <omp.h>

// Tag dispatch for implementation selection
struct SoATag {};
struct AoSTag {};

// Base Dataset structure
struct DatasetBase {
    size_t n_points;
    size_t n_dims;
};

// SoA implementation
struct DatasetSoA : DatasetBase {
    std::vector<float> data; // Single contiguous array for all data

    void resize(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        data.resize(points * dims);
    }

    // Optimized accessors for better cache usage
    float* get_dimension(size_t dim) {
        return &data[dim * n_points];
    }

    const float* get_dimension(size_t dim) const {
        return &data[dim * n_points];
    }

    float& at(size_t dim, size_t point) {
        return data[dim * n_points + point];
    }

    const float& at(size_t dim, size_t point) const {
        return data[dim * n_points + point];
    }
};

// AoS implementation
struct DatasetAoS : DatasetBase {
    std::vector<float> coords;

    void resize(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        coords.resize(points * dims);
    }

    // Optimized accessors
    float* get_point(size_t point) {
        return &coords[point * n_dims];
    }

    const float* get_point(size_t point) const {
        return &coords[point * n_dims];
    }

    float& at(size_t point, size_t dim) {
        return coords[point * n_dims + dim];
    }

    const float& at(size_t point, size_t dim) const {
        return coords[point * n_dims + dim];
    }
};

// Base Centroids structure
struct CentroidsBase {
    size_t k;
    size_t n_dims;
    std::vector<size_t> counts;
};

// SoA implementation
struct CentroidsSoA : CentroidsBase {
    std::vector<float> data;

    void resize(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        data.resize(clusters * dims);
        counts.resize(clusters);
    }

    float* get_dimension(size_t dim) {
        return &data[dim * k];
    }

    const float* get_dimension(size_t dim) const {
        return &data[dim * k];
    }

    float& at(size_t dim, size_t cluster) {
        return data[dim * k + cluster];
    }

    const float& at(size_t dim, size_t cluster) const {
        return data[dim * k + cluster];
    }
};

// AoS implementation
struct CentroidsAoS : CentroidsBase {
    std::vector<float> coords;

    void resize(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        coords.resize(clusters * dims);
        counts.resize(clusters);
    }

    float* get_centroid(size_t cluster) {
        return &coords[cluster * n_dims];
    }

    const float* get_centroid(size_t cluster) const {
        return &coords[cluster * n_dims];
    }

    float& at(size_t cluster, size_t dim) {
        return coords[cluster * n_dims + dim];
    }

    const float& at(size_t cluster, size_t dim) const {
        return coords[cluster * n_dims + dim];
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
    size_t n_points = 1;
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
                data.at(dim_idx, point_idx) = std::stof(line.substr(0, pos));
                line.erase(0, pos + 1);
                dim_idx++;
            }
            data.at(dim_idx, point_idx) = std::stof(line);
            point_idx++;
        }
    } else {
        // AoS reading implementation
        size_t point_idx = 0;
        while (std::getline(file, line)) {
            size_t pos = 0;
            size_t dim_idx = 0;
            while ((pos = line.find(',')) != std::string::npos) {
                data.at(point_idx, dim_idx) = std::stof(line.substr(0, pos));
                line.erase(0, pos + 1);
                dim_idx++;
            }
            data.at(point_idx, dim_idx) = std::stof(line);
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
        for (size_t i = 0; i < k; ++i) {
            size_t idx = dis(gen);
            for (size_t d = 0; d < data.n_dims; ++d) {
                centroids.at(d, i) = data.at(d, idx);
            }
        }
    } else {
        for (size_t i = 0; i < k; ++i) {
            size_t idx = dis(gen);
            for (size_t d = 0; d < data.n_dims; ++d) {
                centroids.at(i, d) = data.at(idx, d);
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
#pragma omp simd reduction(+:dist)
        for (size_t d = 0; d < data.n_dims; ++d) {
            float diff = data.at(d, point_idx) - centroids.at(d, centroid_idx);
            dist += diff * diff;
        }
    } else {
        const float* point = data.get_point(point_idx);
        const float* centroid = centroids.get_centroid(centroid_idx);
#pragma omp simd reduction(+:dist)
        for (size_t d = 0; d < data.n_dims; ++d) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }
    }

    return dist;
}

template<typename Dataset, typename Centroids>
void kmeans_sequential(const Dataset& data, Centroids& centroids,
                      std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_centroid_data(centroids.k * data.n_dims, 0.0f);

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
        std::fill(new_centroid_data.begin(), new_centroid_data.end(), 0.0f);
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0);

        // Update step
        if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
            for (size_t i = 0; i < data.n_points; ++i) {
                int cluster = assignments[i];
                centroids.counts[cluster]++;
                for (size_t d = 0; d < data.n_dims; ++d) {
                    new_centroid_data[d * centroids.k + cluster] += data.at(d, i);
                }
            }
        } else {
            for (size_t i = 0; i < data.n_points; ++i) {
                int cluster = assignments[i];
                centroids.counts[cluster]++;
                const float* point = data.get_point(i);
                for (size_t d = 0; d < data.n_dims; ++d) {
                    new_centroid_data[cluster * data.n_dims + d] += point[d];
                }
            }
        }

        // Normalize centroids
        if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
            for (size_t j = 0; j < centroids.k; ++j) {
                if (centroids.counts[j] > 0) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.at(d, j) = new_centroid_data[d * centroids.k + j] / centroids.counts[j];
                    }
                }
            }
        } else {
            for (size_t j = 0; j < centroids.k; ++j) {
                if (centroids.counts[j] > 0) {
                    float* centroid = centroids.get_centroid(j);
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroid[d] = new_centroid_data[j * data.n_dims + d] / centroids.counts[j];
                    }
                }
            }
        }
    }
}

template<typename Dataset, typename Centroids>
void kmeans_parallel(const Dataset& data, Centroids& centroids, std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_centroid_data(centroids.k * data.n_dims);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        #pragma omp parallel for default(none) shared(data, centroids, assignments)
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

        // Reset accumulators
        std::fill(new_centroid_data.begin(), new_centroid_data.end(), 0.0f);
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0);

        // Update step with thread-local storage
        #pragma omp parallel default(none) shared(data, centroids, assignments, new_centroid_data)
        {
            std::vector<float> local_centroid_data(centroids.k * data.n_dims, 0.0f);
            std::vector<size_t> local_counts(centroids.k, 0);

            if constexpr (std::is_same_v<Dataset, DatasetSoA>) {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < data.n_points; ++i) {
                    int cluster = assignments[i];
                    local_counts[cluster]++;
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        local_centroid_data[d * centroids.k + cluster] += data.at(d, i);
                    }
                }
            } else {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < data.n_points; ++i) {
                    int cluster = assignments[i];
                    local_counts[cluster]++;
                    const float* point = data.get_point(i);
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        local_centroid_data[cluster * data.n_dims + d] += point[d];
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
                        new_centroid_data[d * centroids.k + j] += local_centroid_data[d * centroids.k + j];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        #pragma omp atomic
                        new_centroid_data[j * data.n_dims + d] += local_centroid_data[j * data.n_dims + d];
                    }
                }
            }
        }

        // Normalize centroids
        for (size_t j = 0; j < centroids.k; ++j) {
            if (centroids.counts[j] > 0) {
                if constexpr (std::is_same_v<Centroids, CentroidsSoA>) {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.at(d, j) = new_centroid_data[d * centroids.k + j] / centroids.counts[j];
                    }
                } else {
                    for (size_t d = 0; d < data.n_dims; ++d) {
                        centroids.at(j, d) = new_centroid_data[j * data.n_dims + d] / centroids.counts[j];
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