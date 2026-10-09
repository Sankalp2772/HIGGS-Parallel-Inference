#include <mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <zlib.h>
#include <algorithm>

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

    if (!file)
        return false;

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

    for (int j = 0; j < H2_SIZE; j++) {

        double sum = model.b2[j];

        for (int i = 0; i < H1_SIZE; i++)
            sum += h1[i] * model.w2[i][j];

        h2[j] = relu(sum);
    }

    double output = model.b3;

    for (int i = 0; i < H2_SIZE; i++)
        output += h2[i] * model.w3[i];

    return sigmoid(output);
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {

        if (rank == 0)
            cout << "Usage: mpirun ... ./ai_mpi <dataset.gz> <rows>\n";

        MPI_Finalize();
        return 1;
    }

    string dataset_file = argv[1];
    long long num_rows = stoll(argv[2]);

    Model model;

    if (!load_model("model_weights.txt", model)) {

        cerr << "Rank " << rank
             << ": Cannot load model_weights.txt\n";

        MPI_Finalize();
        return 1;
    }

    /*
       --------------------------------------------------
       MASTER LOADS DATA
       --------------------------------------------------
    */

    vector<array<float, INPUT_SIZE>> all_data;
    vector<int> all_labels;

    if (rank == 0) {

        all_data.reserve(num_rows);
        all_labels.reserve(num_rows);

        gzFile file =
            gzopen(dataset_file.c_str(), "rb");

        if (!file) {

            cerr << "Cannot open dataset.\n";

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char buffer[8192];

        while (
            static_cast<long long>(all_data.size()) < num_rows &&
            gzgets(file, buffer, sizeof(buffer))
        ) {

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

            all_data.push_back(features);
            all_labels.push_back(label);
        }

        gzclose(file);
    }

    /*
       Broadcast number of rows actually loaded.
    */

    long long actual_rows = all_data.size();

    MPI_Bcast(
        &actual_rows,
        1,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    /*
       --------------------------------------------------
       DISTRIBUTION
       --------------------------------------------------
    */

    long long base = actual_rows / size;
    long long remainder = actual_rows % size;

    long long local_count =
        base + (rank < remainder ? 1 : 0);

    vector<array<float, INPUT_SIZE>> local_data(local_count);
    vector<int> local_labels(local_count);

    /*
       Send each rank its portion.
    */

    if (rank == 0) {

        for (int r = 1; r < size; r++) {

            long long start =
                r * base + min<long long>(r, remainder);

            long long count =
                base + (r < remainder ? 1 : 0);

            MPI_Send(
                all_data.data() + start,
                count * INPUT_SIZE,
                MPI_FLOAT,
                r,
                0,
                MPI_COMM_WORLD
            );

            MPI_Send(
                all_labels.data() + start,
                count,
                MPI_INT,
                r,
                1,
                MPI_COMM_WORLD
            );
        }

        copy(
            all_data.begin(),
            all_data.begin() + local_count,
            local_data.begin()
        );

        copy(
            all_labels.begin(),
            all_labels.begin() + local_count,
            local_labels.begin()
        );
    }
    else {

        MPI_Recv(
            local_data.data(),
            local_count * INPUT_SIZE,
            MPI_FLOAT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        MPI_Recv(
            local_labels.data(),
            local_count,
            MPI_INT,
            0,
            1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /*
       --------------------------------------------------
       AI INFERENCE
       --------------------------------------------------
    */

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    long long local_correct = 0;
    double local_probability_sum = 0.0;

    for (long long i = 0; i < local_count; i++) {

        double probability =
            predict(local_data[i], model);

        int prediction =
            probability >= 0.5 ? 1 : 0;

        if (prediction == local_labels[i])
            local_correct++;

        local_probability_sum += probability;
    }

    double end_time = MPI_Wtime();

    double local_time =
        end_time - start_time;

    /*
       --------------------------------------------------
       REDUCE RESULTS
       --------------------------------------------------
    */

    long long global_correct = 0;

    double global_probability_sum = 0.0;

    double global_time = 0.0;

    MPI_Reduce(
        &local_correct,
        &global_correct,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_probability_sum,
        &global_probability_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    /*
       Maximum rank time determines parallel
       execution time.
    */

    MPI_Reduce(
        &local_time,
        &global_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        double accuracy =
            static_cast<double>(global_correct)
            / actual_rows;

        cout << "\n========================================\n";
        cout << "       HIGGS AI MPI INFERENCE\n";
        cout << "========================================\n";

        cout << "MPI processes      : "
             << size << endl;

        cout << "Processed rows     : "
             << actual_rows << endl;

        cout << "Correct predictions: "
             << global_correct << endl;

        cout << "Accuracy           : "
             << accuracy << endl;

        cout << "Average probability: "
             << global_probability_sum
                / actual_rows
             << endl;

        cout << "Inference time     : "
             << global_time
             << " seconds\n";

        cout << "========================================\n";
    }

    MPI_Finalize();

    return 0;
}
