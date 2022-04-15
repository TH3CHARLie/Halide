#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "cmdline.h"

#include "DefaultCostModel.h"
#include "HalideBuffer.h"
#include "NetworkSize.h"

namespace {

using namespace Halide;

using Halide::Runtime::Buffer;
using std::map;
using std::string;
using std::vector;

struct Flags {
    int epochs = 0;
    std::vector<float> rates = {0.1f};
    string initial_weights_path;
    string weights_out_path;
    int num_cores = 32;
    bool reset_weights = false;
    bool randomize_weights = false;
    bool predict_only = false;
    string best_benchmark_path;
    string best_schedule_path;
    string predictions_file;
    bool verbose;
    bool partition_schedules;
    int limit;
    int stage_idx;

    Flags(int argc, char **argv) {
        cmdline::parser a;

        const char *kNoDesc = "";

        constexpr bool kOptional = false;
        a.add<int>("epochs");
        a.add<string>("rates");
        a.add<string>("initial_weights", '\0', kNoDesc, kOptional, "");
        a.add<string>("weights_out");
        a.add<bool>("randomize_weights", '\0', kNoDesc, kOptional, false);
        a.add<bool>("predict_only", '\0', kNoDesc, kOptional, false);
        a.add<int>("num_cores");
        a.add<string>("best_benchmark");
        a.add<string>("best_schedule");
        a.add<string>("predictions_file");
        a.add<bool>("verbose");
        a.add<bool>("partition_schedules");
        a.add<int>("limit");
        a.add<int>("stage_idx");

        a.parse_check(argc, argv);  // exits if parsing fails

        epochs = a.get<int>("epochs");
        rates = parse_floats(a.get<string>("rates"));
        initial_weights_path = a.get<string>("initial_weights");
        weights_out_path = a.get<string>("weights_out");
        randomize_weights = a.exist("randomize_weights") && a.get<bool>("randomize_weights");
        predict_only = a.exist("predict_only") && a.get<bool>("predict_only");
        best_benchmark_path = a.get<string>("best_benchmark");
        best_schedule_path = a.get<string>("best_schedule");
        predictions_file = a.get<string>("predictions_file");
        verbose = a.exist("verbose") && a.get<bool>("verbose");
        partition_schedules = a.exist("partition_schedules") && a.get<bool>("partition_schedules");
        limit = a.get<int>("limit");
        stage_idx = a.get<int>("stage_idx");

        if (epochs <= 0) {
            std::cerr << "--epochs must be specified and > 0.\n";
            std::cerr << a.usage();
            exit(1);
        }
        if ((!initial_weights_path.empty()) == randomize_weights) {
            std::cerr << "You must specify exactly one of --initial_weights or --randomize_weights.\n";
            std::cerr << a.usage();
            exit(1);
        }
        if (weights_out_path.empty()) {
            std::cerr << "--weights_out must be specified.\n";
            std::cerr << a.usage();
            exit(1);
        }
        if (rates.empty()) {
            std::cerr << "--rates cannot be empty.\n";
            std::cerr << a.usage();
            exit(1);
        }
    }

    std::vector<float> parse_floats(const std::string &s) {
        const char *c = s.c_str();
        std::vector<float> v;
        while (isspace(*c)) {
            ++c;
        }
        while (*c) {
            string f;
            while (*c && !isspace(*c)) {
                f += *c++;
            }
            v.push_back(std::atof(f.c_str()));
            while (isspace(*c)) {
                ++c;
            }
        }
        return v;
    }
};

constexpr int kModels = 1;

struct Sample {
    vector<float> runtimes;  // in msec
    vector<vector<float>> stage_runtimes;
    vector<vector<float>> transform_matrices;
    double prediction[kModels];
    // TODO: adapt for kModels
    vector<double> stage_predictions;
    string filename;
    int32_t schedule_id;
    Buffer<float> schedule_features;
};

struct PipelineSample {
    int32_t pipeline_id;
    int32_t num_stages;
    Buffer<float> pipeline_features;
    map<uint64_t, Sample> schedules;
    uint64_t fastest_schedule_hash;
    float fastest_runtime;  // in msec
    uint64_t pipeline_hash;
};

uint64_t hash_floats(uint64_t h, const float *begin, const float *end) {
    while (begin != end) {
        uint32_t bits = *((const uint32_t *)begin);
        // From boost
        h ^= (bits + 0x9e3779b9 + (h << 6) + (h >> 2));
        begin++;
    }
    return h;
}

bool ends_with(const string &str, const string &suffix) {
    if (str.size() < suffix.size()) {
        return false;
    }
    size_t off = str.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); i++) {
        if (str[off + i] != suffix[i]) {
            return false;
        }
    }
    return true;
}

std::string replace_suffix(const std::string& path, const std::string& old_suffix, const std::string& new_suffix) {
    std::string res = path.substr(0, path.find(old_suffix)) + new_suffix;
    return res;
}

string leaf(const string &path) {
    size_t slash_pos = path.rfind('/');
#ifdef _WIN32
    if (slash_pos == string::npos) {
        // Windows is a thing
        slash_pos = path.rfind('\\');
    }
#endif
    if (slash_pos != string::npos) {
        return path.substr(slash_pos + 1);
    } else {
        return path;
    }
}

void update_per_stage_predictions(PipelineSample &sample, size_t batch_size, Halide::Runtime::Buffer<float> &prediction_buffer) {
    auto it = sample.schedules.begin();
    for (size_t i = 0; i < batch_size; ++i) {
        auto &sched = it->second;
        sched.stage_predictions.resize(sample.num_stages);
        for (int j = 0; j < sample.num_stages; ++j) {
            sched.stage_predictions[j] = prediction_buffer(i, j);
        }
        it++;
    }
}

void save_predictions(const map<int, PipelineSample> &samples, const string &filename) {
    std::ostringstream out;
    for (const auto &p : samples) {
        for (const auto &sched : p.second.schedules) {
            if (sched.second.runtimes.empty()) {
                continue;
            }
            out << sched.second.filename << ", " << sched.second.prediction[0] << ", " << sched.second.runtimes[0] << "\n";
        }
    }

    std::ofstream file(filename, std::ios_base::trunc);
    file << out.str();
    file.close();
    assert(!file.fail());

    std::cout << "Predictions saved to: " << filename << "\n";
}

void save_per_stage_predictions(const map<int, PipelineSample> &samples, const string &prefix) {
    std::vector<string> names = {"bilateral_grid", "interpolated", "blury", "blurx", "blurz", "histogram", "histogram.update(0)",
                                 "repeat_edge", "lambda_0"};
    for (size_t i = 0; i < names.size(); ++i) {
        std::ostringstream out;
        for (const auto &p : samples) {
            for (const auto &sched : p.second.schedules) {
                if (sched.second.runtimes.empty()) {
                    continue;
                }
                out << sched.second.filename << ", " << sched.second.stage_predictions[i] << ", " << sched.second.stage_runtimes[0][i] << "\n";
            }
        }
        std::ofstream file(prefix + names[i] + ".txt", std::ios_base::trunc);
        file << out.str();
        file.close();
        assert(!file.fail());
        std::cout << "Predictions for " << names[i] << " saved to: " << (prefix + names[i] + ".txt") << "\n";
    }
}

void dump_per_schedule_cost_terms(const std::vector<float> &features, const std::vector<float> &relus, const int stage_id) {
    const int num_cores = 20;
    int idx = 0;
    float num_realizations = features[idx++];
    float num_productions = features[idx++];
    float points_computed_per_realization = features[idx++];
    std::cout << "points_computed_per_realization unused: " << points_computed_per_realization << "\n";
    float points_computed_per_production = features[idx++];
    std::cout << "points_computed_per_production unused: " << points_computed_per_production << "\n";
    float points_computed_total = features[idx++];
    std::cout << "points_computed_total unused: " << points_computed_total << "\n";
    float points_computed_minimum = features[idx++];
    std::cout << "points_computed_minimum unused: " << points_computed_minimum << "\n";
    float innermost_loop_extent = features[idx++];
    std::cout << "innermost_loop_extent unused: " << innermost_loop_extent << "\n";
    float innermost_pure_loop_extent = features[idx++];
    std::cout << "innermost_pure_loop_extent unused: " << innermost_pure_loop_extent << "\n";
    float unrolled_loop_extent = features[idx++];
    std::cout << "unrolled_loop_extent unused: " << unrolled_loop_extent << "\n";
    float inner_parallelism = features[idx++];
    float outer_parallelism = features[idx++];
    float bytes_at_realization = features[idx++];
    float bytes_at_production = features[idx++];
    float bytes_at_root = features[idx++];
    std::cout << "bytes_at_root unused: " << bytes_at_root << "\n";
    float innermost_bytes_at_realization = features[idx++];
    std::cout << "innermost_bytes_at_realization unused: " << innermost_bytes_at_realization << "\n";
    float innermost_bytes_at_production = features[idx++];
    std::cout << "innermost_bytes_at_production unused: " << innermost_bytes_at_production << "\n";
    float innermost_bytes_at_root = features[idx++];
    std::cout << "innermost_bytes_at_root unused: " << innermost_bytes_at_root << "\n";
    float inlined_calls = features[idx++];
    float unique_bytes_read_per_realization = features[idx++];
    float unique_lines_read_per_realization = features[idx++];
    float allocation_bytes_read_per_realization = features[idx++];
    std::cout << "allocation_bytes_read_per_realization unused: " << allocation_bytes_read_per_realization << "\n";
    float working_set = features[idx++];
    float vector_size = features[idx++];
    float native_vector_size = features[idx++];
    std::cout << "native_vector_size unused: " << native_vector_size << "\n";
    float num_vectors = features[idx++];
    float num_scalars = features[idx++];
    float scalar_loads_per_vector = features[idx++];
    float vector_loads_per_vector = features[idx++];
    float scalar_loads_per_scalar = features[idx++];
    float bytes_at_task = features[idx++];
    float innermost_bytes_at_task = features[idx++];
    float unique_bytes_read_per_vector = features[idx++];
    float unique_lines_read_per_vector = features[idx++];
    float unique_bytes_read_per_task = features[idx++];
    float unique_lines_read_per_task = features[idx++];
    float working_set_at_task = features[idx++];
    std::cout << "working_set_at_task unused: " << working_set_at_task << "\n";
    float working_set_at_production = features[idx++];
    std::cout << "working_set_at_production unused: " << working_set_at_production << "\n";
    float working_set_at_realization = features[idx++];
    std::cout << "working_set_at_realization unused: " << working_set_at_realization << "\n";
    float working_set_at_root = features[idx++];
    std::cout << "working_set_at_root unused: " << working_set_at_root << "\n";

    float compute_cost = inlined_calls == 0 ? (vector_size * num_vectors * relus[0] + num_scalars * relus[1]) : (vector_size * num_vectors * relus[2] + num_scalars * relus[3]);
    float num_tasks = std::max(1.0f, inner_parallelism * outer_parallelism);
    float tasks_per_core = num_tasks / num_cores;
    float idle_core_wastage = std::ceil(tasks_per_core) / std::max(1.0f, tasks_per_core);

    compute_cost *= idle_core_wastage;

    float load_cost = (num_realizations * unique_lines_read_per_realization * relus[5] +
                    num_realizations * unique_bytes_read_per_realization * relus[6] +
                    num_vectors * scalar_loads_per_vector * relus[7] +
                    num_scalars * scalar_loads_per_scalar * relus[8] +
                    num_vectors * vector_loads_per_vector * relus[9] +
                    num_scalars * unique_bytes_read_per_vector * relus[10] +
                    num_vectors * unique_bytes_read_per_vector * relus[11] +
                    num_scalars * unique_lines_read_per_vector * relus[12] +
                    num_vectors * unique_lines_read_per_vector * relus[13] +
                    num_tasks * unique_bytes_read_per_task * relus[14] +
                    num_tasks * unique_lines_read_per_task * relus[15]);

    float lines_written_per_realization = inner_parallelism * (bytes_at_task / std::max(1.0f, innermost_bytes_at_task));

    float alpha = 0.0f;
    // note we ignore one select branch relus[17] here because w (num_stages) will not be 0
    if (inner_parallelism > 1) {
        alpha = relus[16];
    } else if (stage_id == 0) {
        alpha = relus[17];
    } else {
        alpha = relus[18];
    }

    float beta = 0.0f;
    if (inner_parallelism > 1) {
        alpha = relus[19];
    } else if (stage_id == 0) {
        alpha = relus[20];
    } else {
        alpha = relus[21];
    }

    float store_cost = num_realizations * (lines_written_per_realization * alpha +
                                            bytes_at_realization * beta);
    float cost_of_false_sharing =
        (inner_parallelism > 1) ? (relus[22] * (num_vectors + num_scalars) / std::max(1.0f, innermost_bytes_at_task)) : 0.0f;

    store_cost += cost_of_false_sharing;

    float max_threads_hitting_same_page_fault = std::min(inner_parallelism, 4096.0f / std::max(1.0f, innermost_bytes_at_task));

    const float &num_page_faults = bytes_at_production;

    float cost_of_page_faults = (num_page_faults * max_threads_hitting_same_page_fault *
                                inner_parallelism * outer_parallelism * relus[23]);

    store_cost += cost_of_page_faults;

    float cost_of_malloc = relus[24] * num_realizations;

    float cost_of_parallel_launches = num_productions *
        (inner_parallelism > 1 ? relus[25] : 0.0f);

    float cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * relus[26];


    float cost_of_parallelism = cost_of_parallel_tasks + cost_of_parallel_launches;

    float cost_of_working_set = working_set * relus[27];

    store_cost *= 2;

    std::cout << "num_tasks: " << num_tasks << "\n";
    std::cout << "tasks_per_core: " << tasks_per_core << "\n";
    std::cout << "idle_core_wastage: " << idle_core_wastage << "\n";
    std::cout << "lines_written_per_realization: " << lines_written_per_realization << "\n";
    std::cout << "cost_of_false_sharing: " << cost_of_false_sharing << "\n";
    std::cout << "max_threads_hitting_same_page_fault: " << max_threads_hitting_same_page_fault << "\n";
    std::cout << "num_page_faults: " << num_page_faults << "\n";
    std::cout << "cost_of_parallel_launches: " << cost_of_parallel_launches << "\n";
    std::cout << "cost_of_parallel_tasks: " << cost_of_parallel_tasks << "\n";
    std::cout << "compute_cost: " << compute_cost << "\n";
    std::cout << "store_cost: " << store_cost << "\n";
    std::cout << "load_cost: " << load_cost << "\n";
    std::cout << "cost_of_malloc: " << cost_of_malloc << "\n";
    std::cout << "cost_of_parallelism: " << cost_of_parallelism << "\n";
    std::cout << "cost_of_working_set: " << cost_of_working_set << "\n";

}

// NOTE: only work for single-stage debugging
void dump_intermediate_states(PipelineSample &sample, Halide::Runtime::Buffer<float> &relu_output, size_t batch_size, const int stage_id) {
    const std::string schedule_feature_names[] = {
        "num_realizations",
        "num_productions",
        "points_computed_per_realization",
        "points_computed_per_production",
        "points_computed_total",
        "points_computed_minimum",
        "innermost_loop_extent",
        "innermost_pure_loop_extent",
        "unrolled_loop_extent",
        "inner_parallelism",
        "outer_parallelism",
        "bytes_at_realization",
        "bytes_at_production",
        "bytes_at_root",
        "innermost_bytes_at_realization",
        "innermost_bytes_at_production",
        "innermost_bytes_at_root",
        "inlined_calls",
        "unique_bytes_read_per_realization",
        "unique_lines_read_per_realization",
        "allocation_bytes_read_per_realizatio",
        "working_set",
        "vector_size",
        "native_vector_size",
        "num_vectors",
        "num_scalars",
        "scalar_loads_per_vector",
        "vector_loads_per_vector",
        "scalar_loads_per_scalar",
        "bytes_at_task",
        "innermost_bytes_at_task",
        "unique_bytes_read_per_vector",
        "unique_lines_read_per_vector",
        "unique_bytes_read_per_task",
        "unique_lines_read_per_task",
        "working_set_at_task",
        "working_set_at_production",
        "working_set_at_realization",
        "working_set_at_root"
    };
    auto it = sample.schedules.begin();
    for (size_t i = 0; i < batch_size; ++i) {
        Sample &sched = it->second;
        std::cout << "dumping intermediate states for file: " << sched.filename << "\n";
        std::cout << "schedule features\n";
        std::vector<float> features;
        for (int j = 0; j < sched.schedule_features.dim(0).extent(); ++j) {
            std::cout << schedule_feature_names[j] << ": " << sched.schedule_features(j, 0) << "\n";
            features.push_back(sched.schedule_features(j, 0));
        }
        std::cout << "relu outputs\n";
        std::vector<float> relus;
        for (size_t j = 0; j < 32; ++j) {
            std::cout << "relu(" << j << "): " << relu_output(j, 0, i) << "\n";
            relus.push_back(relu_output(j, 0, i));
        }
        dump_per_schedule_cost_terms(features, relus, stage_id);
        it++;
    }
}

std::vector<float> read_metadatafile(const std::string &filename) {
    std::ifstream file(filename);
    vector<float> scratch(10 * 1024 * 1024);
    file.read((char *)(scratch.data()), scratch.size() * sizeof(float));
    file.close();
    return scratch;
}

// Load all the samples, reading filenames from stdin
map<int, PipelineSample> load_samples(const Flags &flags) {
    map<int, PipelineSample> result;
    vector<float> scratch(10 * 1024 * 1024);

    int best = -1;
    float best_runtime = 1e20f;
    string best_path;

    size_t num_read = 0, num_unique = 0;
    while (!std::cin.eof()) {
        string sample_filename;
        std::cin >> sample_filename;
        if (sample_filename.empty()) {
            continue;
        }
        std::cout << "[DEBUG] reading filename: " << sample_filename << "\n";
        if (!ends_with(sample_filename, ".newsample")) {
            std::cout << "Skipping file: " << sample_filename << "\n";
            continue;
        }
        std::string metadata_filename = replace_suffix(sample_filename, ".newsample", ".metadata");

        std::ifstream file(sample_filename);
        file.read((char *)(scratch.data()), scratch.size() * sizeof(float));
        const size_t floats_read = file.gcount() / sizeof(float);
        const size_t num_features = floats_read - 2;
        const size_t features_per_stage = head2_w + (head1_w + 1) * head1_h;
        file.close();
        // Note we do not check file.fail(). The various failure cases
        // are handled below by checking the number of floats read. We
        // expect truncated files if the benchmarking or
        // autoscheduling procedure crashes and want to filter them
        // out with a warning.

        if (floats_read == scratch.size()) {
            std::cout << "Too-large sample: " << sample_filename << " " << floats_read << "\n";
            continue;
        }
        if (num_features % features_per_stage != 0) {
            std::cout << "Truncated sample: " << sample_filename << " " << floats_read << "\n";
            continue;
        }
        const int num_stages = 1;
        const int true_num_stages = num_features / features_per_stage;
        // const size_t num_stages = num_features / features_per_stage;

        int pipeline_id = *((int32_t *)(&scratch[num_features]));
        const int schedule_id = *((int32_t *)(&scratch[num_features + 1]));

        std::vector<float> metadata_scratch = read_metadatafile(metadata_filename);
        const int runtime_size = *((int32_t *)(&metadata_scratch[0]));
        const int ordering_size = *((int32_t *)(&metadata_scratch[1]));
        std::vector<float> stage_runtimes;
        for (int i = 0; i < runtime_size; ++i) {
            if (i != flags.stage_idx) {
                continue;
            }
            stage_runtimes.push_back(metadata_scratch[2 + i]);
            std::cout << "[DEBUG] stage runtime " << metadata_scratch[2 + i] << "\n";
        }

        const float runtime = std::accumulate(stage_runtimes.begin(), stage_runtimes.end(), 0.0f);
        std::cout << "[DEBUG] e2e runtime " << runtime << "\n";
        if (runtime > 100000) {  // Don't try to predict runtime over 100s
            std::cout << "Implausible runtime in ms: " << runtime << "\n";
            continue;
        }
        std::vector<int> transform_matrix;
        for (int i = 0; i < ordering_size * ordering_size; ++i) {
            transform_matrix.push_back(*((int32_t *)(&metadata_scratch[2 + runtime_size + i])));
        }
        // std::cout << "Runtime: " << runtime << "\n";
        if (runtime < best_runtime) {
            best_runtime = runtime;
            best = schedule_id;
            best_path = sample_filename;
        }

        PipelineSample &ps = result[pipeline_id];

        if (ps.pipeline_features.data() == nullptr) {
            ps.pipeline_id = pipeline_id;
            ps.num_stages = (int)num_stages;
            ps.pipeline_features = Buffer<float>(head1_w, head1_h, num_stages);
            ps.fastest_runtime = 1e30f;
            for (int i = 0; i < true_num_stages; i++) {
                if (i != flags.stage_idx) {
                    continue;
                }
                for (int x = 0; x < head1_w; x++) {
                    for (int y = 0; y < head1_h; y++) {
                        float f = scratch[i * features_per_stage + (x + 1) * 7 + y + head2_w];
                        if (f < 0 || std::isnan(f)) {
                            std::cout << "Negative or NaN pipeline feature: " << x << " " << y << " " << i << " " << f << "\n";
                        }
                        // ps.pipeline_features(x, y, i) = f;
                        // std::cout << "[DEBUG] reading pipeline feature: " << f << "\n";
                        ps.pipeline_features(x, y, 0) = f;
                    }
                }
            }

            ps.pipeline_hash = hash_floats(0, ps.pipeline_features.begin(), ps.pipeline_features.end());
        }

        uint64_t schedule_hash = 0;
        for (int i = 0; i < true_num_stages; i++) {
            if (i != flags.stage_idx) {
                continue;
            }
            schedule_hash =
                hash_floats(schedule_hash,
                            &scratch[i * features_per_stage],
                            &scratch[i * features_per_stage + head2_w]);
        }

        auto it = ps.schedules.find(schedule_hash);
        if (it != ps.schedules.end()) {
            // Keep the smallest runtime at the front
            float best = it->second.runtimes[0];
            if (runtime < best) {
                it->second.runtimes.push_back(best);
                it->second.runtimes[0] = runtime;
                it->second.filename = sample_filename;
                vector<float> stage_runtime_buffer;
                for (int i = 0; i < ordering_size; ++i) {
                    if (i < runtime_size) {
                        stage_runtime_buffer.push_back(stage_runtimes[i]);
                    } else {
                        stage_runtime_buffer.push_back(0.0f);
                    }
                }
                vector<float> transform_matrix_buffer;
                for (int i = 0; i < ordering_size; ++i) {
                    for (int j = 0; j < ordering_size; ++j) {
                            transform_matrix_buffer.push_back(transform_matrix[i * ordering_size + j]);
                    }
                }
                it->second.stage_runtimes.push_back(it->second.stage_runtimes[0]);
                it->second.stage_runtimes[0] = stage_runtime_buffer;
                it->second.transform_matrices.push_back(it->second.transform_matrices[0]);
                it->second.transform_matrices[0] = transform_matrix_buffer;
            } else {
                it->second.runtimes.push_back(runtime);
            }
            if (runtime < ps.fastest_runtime) {
                ps.fastest_runtime = runtime;
                ps.fastest_schedule_hash = schedule_hash;
            }
        } else {
            Sample sample;
            sample.filename = sample_filename;
            sample.runtimes.push_back(runtime);
            for (double &d : sample.prediction) {
                d = 0.0;
            }
            sample.schedule_id = schedule_id;
            sample.schedule_features = Buffer<float>(head2_w, num_stages);
            vector<float> stage_runtime_buffer;
            for (int i = 0; i < ordering_size; ++i) {
                if (i < runtime_size) {
                    stage_runtime_buffer.push_back(stage_runtimes[i]);
                } else {
                    stage_runtime_buffer.push_back(0.0f);
                }
            }
            vector<float> transform_matrix_buffer;
            for (int i = 0; i < ordering_size; ++i) {
                for (int j = 0; j < ordering_size; ++j) {
                        transform_matrix_buffer.push_back(transform_matrix[i * ordering_size + j]);
                }
            }
            sample.stage_runtimes.push_back(stage_runtime_buffer);
            sample.transform_matrices.push_back(transform_matrix_buffer);

            bool ok = true;
            for (int i = 0; i < true_num_stages; i++) {
                if (i != flags.stage_idx) {
                    continue;
                }
                for (int x = 0; x < head2_w; x++) {
                    float f = scratch[i * features_per_stage + x];
                    if (f < 0 || f > 1e14 || std::isnan(f)) {
                        std::cout << "Negative or implausibly large schedule feature: " << i << " " << x << " " << f << "\n";
                        // Something must have overflowed
                        ok = false;
                    }
                    // sample.schedule_features(x, i) = f;
                    sample.schedule_features(x, 0) = f;
                    // std::cout << "[DEBUG] reading schedule feature: " << f << "\n";
                }
                /*
                if (sample.schedule_features(0, i) != sample.schedule_features(1, i)) {
                    std::cout << "Rejecting sliding window schedule for now\n";
                    ok = false;
                }
                */
            }
            if (ok) {
                if (runtime < ps.fastest_runtime) {
                    ps.fastest_runtime = runtime;
                    ps.fastest_schedule_hash = schedule_hash;
                }
                ps.schedules.emplace(schedule_hash, std::move(sample));
                num_unique++;
            }
        }
        num_read++;

        if (num_read % 10000 == 0) {
            std::cout << "Samples loaded: " << num_read << " (" << num_unique << " unique)\n";
        }
        // int aaaaaaaaaa;
        // std::cin >> aaaaaaaaaa;
    }

    // Check the noise level
    for (const auto &pipe : result) {
        double variance_sum = 0;
        size_t count = 0;
        // Compute the weighted average of variances across all samples
        for (const auto &p : pipe.second.schedules) {
            if (p.second.runtimes.empty()) {
                std::cerr << "Empty runtimes for schedule: " << p.first << "\n";
                abort();
            }
            std::cout << "Unique sample: " << leaf(p.second.filename) << " : " << p.second.runtimes.size() << " " << p.second.runtimes[0] << "\n";
            if (p.second.runtimes.size() > 1) {
                // Compute variance from samples
                double mean = 0;
                for (float f : p.second.runtimes) {
                    mean += f;
                }
                mean /= p.second.runtimes.size();
                double variance = 0;
                for (float f : p.second.runtimes) {
                    f -= mean;
                    variance += f * f;
                }
                variance_sum += variance;
                count += p.second.runtimes.size() - 1;
            }
        }
        if (count > 0) {
            double stddev = std::sqrt(variance_sum / count);
            std::cout << "Noise level: " << stddev << "\n";
        }
    }

    std::cout << "Distinct pipelines: " << result.size() << "\n";

    std::ostringstream o;
    o << "Best runtime is " << best_runtime << " msec, from schedule id " << best << " in file " << best_path << "\n";
    std::cout << o.str();
    if (!flags.best_benchmark_path.empty()) {
        std::ofstream f(flags.best_benchmark_path, std::ios_base::trunc);
        f << o.str();
        f.close();
        assert(!f.fail());
    }
    if (!flags.best_schedule_path.empty()) {
        // best_path points to a .sample file; look for a .schedule.h file in the same dir
        size_t dot = best_path.rfind('.');
        assert(dot != string::npos && best_path.substr(dot) == ".newsample");
        string schedule_file = best_path.substr(0, dot) + ".schedule.h";
        std::ifstream src(schedule_file);
        std::ofstream dst(flags.best_schedule_path);
        dst << src.rdbuf();
        assert(!src.fail());
        assert(!dst.fail());
    }

    return result;
}

}  // namespace

// TODO: fix this constant
const int ordering_size = 9;

int main(int argc, char **argv) {
    Flags flags(argc, argv);

    auto samples = load_samples(flags);
    std::cout << "Sample loading finished" << std::endl;
    bool predict_only = flags.predict_only;

    // Iterate through the pipelines
    vector<std::unique_ptr<DefaultCostModel>> tpp;
    for (int i = 0; i < kModels; i++) {
        tpp.emplace_back(make_default_cost_model(flags.initial_weights_path, flags.weights_out_path, flags.randomize_weights));
    }

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);

    auto seed = time(nullptr);
    std::mt19937 rng((uint32_t)seed);

    std::cout << "Iterating over " << samples.size() << " samples using seed = " << seed << "\n";
    decltype(samples) validation_set;
    uint64_t unique_schedules = 0;
    if (samples.size() > 16) {
        for (const auto &p : samples) {
            unique_schedules += p.second.schedules.size();
            // Whether or not a pipeline is part of the validation set
            // can't be a call to rand. It must be a fixed property of a
            // hash of some aspect of it.  This way you don't accidentally
            // do a training run where a validation set member was in the
            // training set of a previous run. The id of the fastest
            // schedule will do as a hash.
            if ((p.second.pipeline_hash & 7) == 0) {
                validation_set.insert(p);
            }
        }

        for (const auto &p : validation_set) {
            samples.erase(p.first);
        }
    }

    std::cout << "Number of unique schedules: " << unique_schedules << "\n";

    std::cout << "Start training" << std::endl;
    for (float learning_rate : flags.rates) {
        float loss_sum[kModels] = {0}, loss_sum_counter[kModels] = {0};
        float correct_ordering_rate_sum[kModels] = {0};
        float correct_ordering_rate_count[kModels] = {0};
        float v_correct_ordering_rate_sum[kModels] = {0};
        float v_correct_ordering_rate_count[kModels] = {0};

        for (int e = 0; e < flags.epochs; e++) {
            int counter = 0;

            float worst_miss = 0;
            uint64_t worst_miss_pipeline_id = 0;
            uint64_t worst_miss_schedule_id = 0;

            struct Inversion {
                int pipeline_id;
                string f1, f2;
                float p1, p2;
                float r1, r2;
                float badness = 0;
            } worst_inversion;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (int model = 0; model < kModels; model++) {
                for (int train = 0; train < 2; train++) {
                    auto &tp = tpp[model];

                    for (auto &p : train ? samples : validation_set) {
                        if (kModels > 1 && rng() & 1) {
                            continue;  // If we are training multiple kModels, allow them to diverge.
                        }
                        // if (p.second.schedules.size() < 8) {
                        //     continue;
                        // }
                        tp->reset();
                        tp->set_pipeline_features(p.second.pipeline_features, flags.num_cores);

                        size_t batch_size = std::min((size_t)1024, p.second.schedules.size());
                        std::cout << "batch_size: " << batch_size << "\n";
                        size_t fastest_idx = 0;
                        Halide::Runtime::Buffer<float> runtimes(batch_size);
                        Halide::Runtime::Buffer<float> stage_runtimes(batch_size, ordering_size);
                        Halide::Runtime::Buffer<float> transformed_stage_runtimes(batch_size, ordering_size);
                        Halide::Runtime::Buffer<float> transform_matrices(batch_size, ordering_size, ordering_size);
                        size_t first = 0;
                        if (p.second.schedules.size() > 1024) {
                            first = rng() % (p.second.schedules.size() - 1024);
                        }
                        Halide::Runtime::Buffer<float> stage_predictions(batch_size, ordering_size);
                        auto it = p.second.schedules.begin();
                        std::advance(it, first);
                        for (size_t j = 0; j < batch_size; j++) {
                            auto &sched = it->second;
                            Halide::Runtime::Buffer<float> buf;
                            tp->enqueue(p.second.num_stages, &buf, &sched.prediction[model]);
                            runtimes(j) = sched.runtimes[0];
                            for (size_t k = 0; k < ordering_size; ++k) {
                                stage_runtimes(j, k) = sched.stage_runtimes[0][k];
                            }
                            for (size_t k = 0; k < ordering_size; ++k) {
                                for (size_t l = 0; l < ordering_size; ++l) {
                                    transform_matrices(j, k, l) = sched.transform_matrices[0][k * ordering_size + l];
                                }
                            }
                            if (runtimes(j) < runtimes(fastest_idx)) {
                                fastest_idx = j;
                            }
                            buf.copy_from(sched.schedule_features);
                            it++;
                        }

                        float loss = 0.0f;
                        if (train & !predict_only) {
                            loss = tp->backprop(runtimes, stage_runtimes, transform_matrices, learning_rate);
                            assert(!std::isnan(loss));
                            loss_sum[model] += loss;
                            loss_sum_counter[model]++;

                            auto it = p.second.schedules.begin();
                            std::advance(it, first);
                            for (size_t j = 0; j < batch_size; j++) {
                                auto &sched = it->second;
                                float m = sched.runtimes[0] / (sched.prediction[model] + 1e-10f);
                                if (m > worst_miss) {
                                    worst_miss = m;
                                    worst_miss_pipeline_id = p.first;
                                    worst_miss_schedule_id = it->first;
                                }
                                it++;
                            }
                        } else {
                            Halide::Runtime::Buffer<float> stage_predictions_output(batch_size, ordering_size);
                            Halide::Runtime::Buffer<float> transformed_stage_predictions_output(batch_size, ordering_size);
                            Halide::Runtime::Buffer<float> relu_output(32, 1, batch_size);
                            tp->evaluate_costs_with_stage_runtimes(transform_matrices, relu_output);
                            update_per_stage_predictions(p.second, batch_size, transformed_stage_predictions_output);
                            dump_intermediate_states(p.second, relu_output, batch_size, flags.stage_idx);
                            // for (size_t ii = 0; ii < batch_size; ++ii) {
                            //     std::cout << "matrix " << ii << "\n";
                            //     for (int jj = 0; jj < ordering_size; ++jj) {
                            //         for (int kk = 0; kk < ordering_size; ++kk) {
                            //             std::cout << transform_matrices(ii, jj, kk) << " ";
                            //         }
                            //         std::cout << '\n';
                            //     }
                            // }
                        }

                        if (true) {
                            int good = 0, bad = 0;
                            for (auto &sched : p.second.schedules) {
                                auto &ref = p.second.schedules[p.second.fastest_schedule_hash];
                                if (sched.second.prediction[model] == 0) {
                                    continue;
                                }
                                assert(sched.second.runtimes[0] >= ref.runtimes[0]);
                                float runtime_ratio = sched.second.runtimes[0] / ref.runtimes[0];
                                if (runtime_ratio <= 1.3f) {
                                    continue;  // Within 30% of the runtime of the best
                                }
                                if (sched.second.prediction[model] >= ref.prediction[model]) {
                                    good++;
                                } else {
                                    if (train) {
                                        float badness = (sched.second.runtimes[0] - ref.runtimes[0]) * (ref.prediction[model] - sched.second.prediction[model]);
                                        badness /= (ref.runtimes[0] * ref.runtimes[0]);
                                        if (badness > worst_inversion.badness) {
                                            worst_inversion.pipeline_id = p.first;
                                            worst_inversion.badness = badness;
                                            worst_inversion.r1 = ref.runtimes[0];
                                            worst_inversion.r2 = sched.second.runtimes[0];
                                            worst_inversion.p1 = ref.prediction[model];
                                            worst_inversion.p2 = sched.second.prediction[model];
                                            worst_inversion.f1 = ref.filename;
                                            worst_inversion.f2 = sched.second.filename;
                                        }
                                    }
                                    bad++;
                                }
                            }
                            if (train) {
                                correct_ordering_rate_sum[model] += good;
                                correct_ordering_rate_count[model] += good + bad;
                            } else {
                                v_correct_ordering_rate_sum[model] += good;
                                v_correct_ordering_rate_count[model] += good + bad;
                            }
                        }
                    }
                }

                counter++;
            }
            std::cout << "Epoch: " << e << " ";
            std::cout << "Loss: ";
            for (int model = 0; model < kModels; model++) {
                std::cout << loss_sum[model] / loss_sum_counter[model] << " ";
                loss_sum[model] *= 0.9f;
                loss_sum_counter[model] *= 0.9f;
            }
            if (kModels > 1) {
                std::cout << "\n";
            }
            std::cout << " Rate: ";
            int best_model = 0;
            float best_rate = 0;
            for (int model = 0; model < kModels; model++) {
                float rate = correct_ordering_rate_sum[model] / correct_ordering_rate_count[model];
                std::cout << rate << " ";
                correct_ordering_rate_sum[model] *= 0.9f;
                correct_ordering_rate_count[model] *= 0.9f;

                rate = v_correct_ordering_rate_sum[model] / v_correct_ordering_rate_count[model];
                if (rate < best_rate) {
                    best_model = model;
                    best_rate = rate;
                }
                std::cout << rate << " ";
                v_correct_ordering_rate_sum[model] *= 0.9f;
                v_correct_ordering_rate_count[model] *= 0.9f;
            }

            if (kModels > 1) {
                std::cout << "\n";
            }
            if (samples.count(worst_miss_pipeline_id)) {
                std::cout << " Worst: " << worst_miss << " " << leaf(samples[worst_miss_pipeline_id].schedules[worst_miss_schedule_id].filename) << "\n";
                // samples[worst_miss_pipeline_id].schedules.erase(worst_miss_schedule_id);
            } else {
                std::cout << "\n";
            }

            if (worst_inversion.badness > 0) {
                std::cout << "Worst inversion:\n"
                          << leaf(worst_inversion.f1) << " predicted: " << worst_inversion.p1 << " actual: " << worst_inversion.r1 << "\n"
                          << leaf(worst_inversion.f2) << " predicted: " << worst_inversion.p2 << " actual: " << worst_inversion.r2 << "\n";
                if (samples.size() > 50000) {
                    // For robustness during training on large numbers
                    // of random pipelines, we discard poorly
                    // performing samples from the training set
                    // only. Some of them are weird degenerate
                    // pipelines.
                    samples.erase(worst_inversion.pipeline_id);
                }
            }

            tpp[best_model]->save_weights();

            if (loss_sum[best_model] < 1e-5f) {
                save_predictions(samples, flags.predictions_file);
                // save_per_stage_predictions(samples, "per_stage_");
                std::cout << "Zero loss, returning early\n";
                return 0;
            }
        }
    }

    // tpp.save_weights();
    if (predict_only) {
        save_predictions(samples, flags.predictions_file);
        // save_predictions(validation_set, flags.predictions_file + "_validation_set");
    }
    return 0;
}
