
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int INPUTS = 28;
constexpr int H1 = 32;
constexpr int H2 = 16;
constexpr int THREADS = 256;

struct Model {
    float mean[INPUTS], scale[INPUTS];
    float w1[INPUTS * H1], b1[H1];
    float w2[H1 * H2], b2[H2];
    float w3[H2], b3;
};

bool expectToken(std::istream& in, const std::string& expected) {
    std::string token;
    if (!(in >> token) || token != expected) {
        std::cerr << "Model format error: expected " << expected
                  << ", got " << token << '\n';
        return false;
    }
    return true;
}

bool readValues(std::istream& in, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!(in >> dst[i])) return false;
    }
    return true;
}

bool loadModel(const char* path, Model& m) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open " << path << '\n';
        return false;
    }

    std::string line;
    std::getline(in, line); // # StandardScaler

    if (!expectToken(in, "SCALER_MEAN") ||
        !readValues(in, m.mean, INPUTS) ||
        !expectToken(in, "SCALER_SCALE") ||
        !readValues(in, m.scale, INPUTS) ||
        !expectToken(in, "LAYER_1_WEIGHTS"))
        return false;

    int rows, cols;
    if (!(in >> rows >> cols) || rows != INPUTS || cols != H1 ||
        !readValues(in, m.w1, INPUTS * H1) ||
        !expectToken(in, "LAYER_1_BIASES"))
        return false;

    int n;
    if (!(in >> n) || n != H1 || !readValues(in, m.b1, H1) ||
        !expectToken(in, "LAYER_2_WEIGHTS"))
        return false;

    if (!(in >> rows >> cols) || rows != H1 || cols != H2 ||
        !readValues(in, m.w2, H1 * H2) ||
        !expectToken(in, "LAYER_2_BIASES"))
        return false;

    if (!(in >> n) || n != H2 || !readValues(in, m.b2, H2) ||
        !expectToken(in, "LAYER_3_WEIGHTS"))
        return false;

    if (!(in >> rows >> cols) || rows != H2 || cols != 1 ||
        !readValues(in, m.w3, H2) ||
        !expectToken(in, "LAYER_3_BIASES") ||
        !(in >> n) || n != 1 || !(in >> m.b3))
        return false;

    return true;
}

__device__ float sigmoidf_safe(float x) {
    // Stable sigmoid, avoiding overflow for large negative inputs.
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

__global__ void inferenceKernel(
    const float* X, const int* labels,
    const float* mean, const float* scale,
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    const float* w3, float b3,
    int n, unsigned long long* correct,
    double* probabilitySum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float a1[H1];
    float a2[H2];

    // StandardScaler: (x - mean) / scale
    for (int j = 0; j < H1; ++j) {
        float z = b1[j];
        for (int k = 0; k < INPUTS; ++k) {
            float x = (X[i * INPUTS + k] - mean[k]) / scale[k];
            z += x * w1[k * H1 + j];
        }
        a1[j] = fmaxf(z, 0.0f);
    }

    // Dense 32 -> 16 + ReLU
    for (int j = 0; j < H2; ++j) {
        float z = b2[j];
        for (int k = 0; k < H1; ++k)
            z += a1[k] * w2[k * H2 + j];
        a2[j] = fmaxf(z, 0.0f);
    }

    // Dense 16 -> 1 + sigmoid
    float z = b3;
    for (int k = 0; k < H2; ++k)
        z += a2[k] * w3[k];

    float probability = sigmoidf_safe(z);
    int prediction = probability >= 0.5f ? 1 : 0;

    if (prediction == labels[i])
        atomicAdd(correct, 1ULL);

    atomicAdd(probabilitySum, (double)probability);
}

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)       \
                  << " at line " << __LINE__ << '\n';              \
        std::exit(1);                                              \
    }                                                              \
} while (0)

template <typename T>
T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p),
                          count * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    const char* dataPath = "HIGGS_1M.csv";
    const char* modelPath = "model_weights.txt";
    int limit = 10000; // Safe initial test size.

    if (argc >= 2) dataPath = argv[1];
    if (argc >= 3) limit = std::atoi(argv[2]);
    if (argc >= 4) modelPath = argv[3];

    if (limit <= 0) {
        std::cerr << "Row count must be positive.\n";
        return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA GPU detected.\n";
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << '\n';

    Model model{};
    if (!loadModel(modelPath, model)) {
        std::cerr << "Failed to load neural-network weights.\n";
        return 1;
    }

    // Read and parse before timing GPU inference.
    std::ifstream file(dataPath);
    if (!file) {
        std::cerr << "Cannot open dataset: " << dataPath << '\n';
        return 1;
    }

    std::vector<float> X;
    std::vector<int> labels;
    X.reserve(static_cast<size_t>(limit) * INPUTS);
    labels.reserve(limit);

    std::string line;
    while (static_cast<int>(labels.size()) < limit &&
           std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string field;
        float values[INPUTS + 1];
        bool valid = true;

        for (int j = 0; j < INPUTS + 1; ++j) {
            if (!std::getline(ss, field, ',')) {
                valid = false;
                break;
            }
            try {
                size_t used = 0;
                values[j] = std::stof(field, &used);
                if (used != field.size() || !std::isfinite(values[j]))
                    valid = false;
            } catch (...) {
                valid = false;
            }
            if (!valid) break;
        }

        if (!valid) {
            std::cerr << "Invalid CSV row " << labels.size() + 1 << '\n';
            return 1;
        }

        labels.push_back(values[0] >= 0.5f ? 1 : 0);
        for (int j = 0; j < INPUTS; ++j)
            X.push_back(values[j + 1]);
    }

    const int n = static_cast<int>(labels.size());
    if (n == 0) {
        std::cerr << "No dataset rows were read.\n";
        return 1;
    }

    std::cout << "Rows loaded: " << n << '\n';

    float *dX, *dMean, *dScale, *dw1, *db1, *dw2, *db2, *dw3;
    int* dLabels;
    unsigned long long* dCorrect;
    double* dProbabilitySum;

    dX = deviceAlloc<float>(X.size());
    dLabels = deviceAlloc<int>(labels.size());
    dMean = deviceAlloc<float>(INPUTS);
    dScale = deviceAlloc<float>(INPUTS);
    dw1 = deviceAlloc<float>(INPUTS * H1);
    db1 = deviceAlloc<float>(H1);
    dw2 = deviceAlloc<float>(H1 * H2);
    db2 = deviceAlloc<float>(H2);
    dw3 = deviceAlloc<float>(H2);
    dCorrect = deviceAlloc<unsigned long long>(1);
    dProbabilitySum = deviceAlloc<double>(1);

    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLabels, labels.data(),
                         labels.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMean, model.mean, sizeof(model.mean),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dScale, model.scale, sizeof(model.scale),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw1, model.w1, sizeof(model.w1),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, model.b1, sizeof(model.b1),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2, model.w2, sizeof(model.w2),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, model.b2, sizeof(model.b2),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw3, model.w3, sizeof(model.w3),
                         cudaMemcpyHostToDevice));

    unsigned long long zeroCorrect = 0;
    double zeroProbability = 0.0;
    CUDA_CHECK(cudaMemcpy(dCorrect, &zeroCorrect, sizeof(zeroCorrect),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dProbabilitySum, &zeroProbability,
                         sizeof(zeroProbability), cudaMemcpyHostToDevice));

    int blocks = (n + THREADS - 1) / THREADS;

    // Warm-up launch; exclude startup overhead from the timed inference.
    inferenceKernel<<<blocks, THREADS>>>(
        dX, dLabels, dMean, dScale, dw1, db1, dw2, db2, dw3,
        model.b3, n, dCorrect, dProbabilitySum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dCorrect, &zeroCorrect, sizeof(zeroCorrect),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dProbabilitySum, &zeroProbability,
                         sizeof(zeroProbability), cudaMemcpyHostToDevice));

    auto start = std::chrono::steady_clock::now();

    inferenceKernel<<<blocks, THREADS>>>(
        dX, dLabels, dMean, dScale, dw1, db1, dw2, db2, dw3,
        model.b3, n, dCorrect, dProbabilitySum);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto stop = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(&zeroCorrect, dCorrect, sizeof(zeroCorrect),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&zeroProbability, dProbabilitySum,
                         sizeof(zeroProbability), cudaMemcpyDeviceToHost));

    double seconds =
        std::chrono::duration<double>(stop - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Correct predictions: " << zeroCorrect << '\n';
    std::cout << "Accuracy: "
              << static_cast<double>(zeroCorrect) / n << '\n';
    std::cout << "Average probability: " << zeroProbability / n << '\n';
    std::cout << "GPU inference time (s): " << seconds << '\n';
    std::cout << "Rows per second: " << n / seconds << '\n';

    cudaFree(dX);
    cudaFree(dLabels);
    cudaFree(dMean);
    cudaFree(dScale);
    cudaFree(dw1);
    cudaFree(db1);
    cudaFree(dw2);
    cudaFree(db2);
    cudaFree(dw3);
    cudaFree(dCorrect);
    cudaFree(dProbabilitySum);

    return 0;
}
