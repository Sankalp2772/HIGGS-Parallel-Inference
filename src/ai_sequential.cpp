#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <string>
#include <zlib.h>

using namespace std;

constexpr int INPUT_SIZE = 28;
constexpr int H1_SIZE = 32;
constexpr int H2_SIZE = 16;

struct Model {
    array<double, INPUT_SIZE> mean{};
    array<double, INPUT_SIZE> scale{};

    double w1[INPUT_SIZE][H1_SIZE];
    double b1[H1_SIZE];

    double w2[H1_SIZE][H2_SIZE];
    double b2[H2_SIZE];

    double w3[H2_SIZE];
    double b3;
};

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

bool load_model(const string& filename, Model& model) {

    ifstream file(filename);

    if (!file) {
        cerr << "ERROR: Cannot open model file: "
             << filename << endl;
        return false;
    }

    string line;

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < INPUT_SIZE; i++)
        file >> model.mean[i];

    file >> line;

    for (int i = 0; i < INPUT_SIZE; i++)
        file >> model.scale[i];

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < H1_SIZE; j++)
            file >> model.w1[i][j];

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < H1_SIZE; i++)
        file >> model.b1[i];

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < H1_SIZE; i++)
        for (int j = 0; j < H2_SIZE; j++)
            file >> model.w2[i][j];

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < H2_SIZE; i++)
        file >> model.b2[i];

    getline(file, line);
    getline(file, line);

    for (int i = 0; i < H2_SIZE; i++)
        file >> model.w3[i];

    getline(file, line);
    getline(file, line);

    file >> model.b3;

    return true;
}

double predict(
    const array<float, INPUT_SIZE>& input,
    const Model& model
) {

    double h1[H1_SIZE];
    double h2[H2_SIZE];

    // Standardization + Layer 1
    for (int j = 0; j < H1_SIZE; j++) {

        double sum = model.b1[j];

        for (int i = 0; i < INPUT_SIZE; i++) {

            double x =
                (static_cast<double>(input[i])
                - model.mean[i])
                / model.scale[i];

            sum += x * model.w1[i][j];
        }

        h1[j] = relu(sum);
    }

    // Layer 2
    for (int j = 0; j < H2_SIZE; j++) {

        double sum = model.b2[j];

        for (int i = 0; i < H1_SIZE; i++)
            sum += h1[i] * model.w2[i][j];

        h2[j] = relu(sum);
    }

    // Output layer
    double output = model.b3;

    for (int i = 0; i < H2_SIZE; i++)
        output += h2[i] * model.w3[i];

    return sigmoid(output);
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Usage: ./ai_sequential <dataset.gz> <rows>\n";
        return 1;
    }

    string dataset_file = argv[1];
    long long num_rows = stoll(argv[2]);

    Model model;

    if (!load_model("model_weights.txt", model))
        return 1;

    /*
       --------------------------------------------------
       PHASE 1: Load dataset
       --------------------------------------------------
       This phase is NOT included in inference timing.
    */

    vector<array<float, INPUT_SIZE>> data;
    vector<int> labels;

    data.reserve(num_rows);
    labels.reserve(num_rows);

    gzFile file = gzopen(dataset_file.c_str(), "rb");

    if (!file) {
        cerr << "ERROR: Cannot open dataset: "
             << dataset_file << endl;
        return 1;
    }

    char buffer[8192];

    while (static_cast<long long>(data.size()) < num_rows &&
           gzgets(file, buffer, sizeof(buffer))) {

        string line(buffer);
        stringstream ss(line);

        string value;

        if (!getline(ss, value, ','))
            continue;

        int label = static_cast<int>(stof(value));

        array<float, INPUT_SIZE> features;

        bool valid = true;

        for (int i = 0; i < INPUT_SIZE; i++) {

            if (!getline(ss, value, ',')) {
                valid = false;
                break;
            }

            features[i] = stof(value);
        }

        if (!valid)
            continue;

        data.push_back(features);
        labels.push_back(label);
    }

    gzclose(file);

    /*
       --------------------------------------------------
       PHASE 2: Sequential AI inference
       --------------------------------------------------
       This is the measured region.
       --------------------------------------------------
    */

    long long correct = 0;
    double probability_sum = 0.0;

    cout << "========================================\n";
    cout << "    HIGGS AI SEQUENTIAL INFERENCE\n";
    cout << "========================================\n";

    cout << "Rows             : " << data.size() << endl;
    cout << "Model            : 28 -> 32 -> 16 -> 1\n";

    auto start = chrono::high_resolution_clock::now();

    for (long long i = 0;
         i < static_cast<long long>(data.size());
         i++) {

        double probability =
            predict(data[i], model);

        int prediction =
            probability >= 0.5 ? 1 : 0;

        if (prediction == labels[i])
            correct++;

        probability_sum += probability;
    }

    auto end = chrono::high_resolution_clock::now();

    double elapsed =
        chrono::duration<double>(end - start).count();

    double accuracy =
        static_cast<double>(correct)
        / data.size();

    cout << "\n========================================\n";
    cout << "RESULTS\n";
    cout << "========================================\n";

    cout << "Processed rows      : "
         << data.size() << endl;

    cout << "Correct predictions : "
         << correct << endl;

    cout << "Accuracy            : "
         << accuracy << endl;

    cout << "Average probability : "
         << probability_sum / data.size()
         << endl;

    cout << "Inference time      : "
         << elapsed << " seconds\n";

    cout << "========================================\n";

    return 0;
}
