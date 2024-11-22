# A Comparative Analysis of AoS vs SoA in Parallel K-Means Implementation with OpenMP
This repository presents a high-performance C++ implementation of the K-Means clustering algorithm, optimized for parallel execution using OpenMP. The project specifically focuses on comparing Array of Structures (AoS) versus Structure of Arrays (SoA) approaches in terms of performance and scalability.

## Prerequisites
The project requires the following components:
- A C++ compiler with OpenMP support (such as g++)
- OpenMP library
  - Pre-installed on most Linux distributions
  - For MacOS users, the library can be installed via package managers like Homebrew

## Setup and Usage
1. Clone the repository:
  ```bash
  git clone lorenzo-27/kmeans-openmp
  cd kmeans-openmp
  ```
2. Configure the algorithm parameters:
   - Open `kmeans_config.py`
   - Adjust the clustering parameters according to your requirements
3. Compile the project:
   - If using CLion, the build process is automatically handled
   - For manual compilation, ensure you create a `cmake-build-release` directory or update the executable path in `kmeans.py`
<details>
  <summary>MacOs Specific Configuration</summary>
  For MacOS users, additional CMake configuration is required. Add the following to your CMakeLists.txt:
  
  ```cmake
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")

  include_directories(/opt/homebrew/opt/libomp/include)
  link_directories(/opt/homebrew/opt/libomp/lib)
  ```
This configuration has been tested and used with MacOS Ventura; it may differ for other MacOS versions.</details>
</details>

> [!NOTE]
> Upon execution, the program automatically creates two directories:
> - data/: Contains generated datasets
> - results/: Stores performance plots and analysis tables

## Documentation
For a comprehensive understanding of the implementation and performance analysis, please refer to our detailed technical report available here. The report includes:
- Implementation details
- Performance benchmarks
- Experimental results and analysis

> [!IMPORTANT]
> The entire implementation is consolidated in main.cpp to optimize parallel performance, as Object-Oriented Programming patterns can introduce overhead in parallel computing contexts.
## License
This project is licensed under the <a href="https://github.com/DavideDelBimbo/K-Means-OpenMP/blob/main/LICENSE" target="_blank">MIT</a> License.
