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

#include "cmdline.h"

#include "HalideBuffer.h"
#include "NetworkSize.h"

namespace {

using namespace Halide;

using Halide::Runtime::Buffer;
using std::map;
using std::string;
using std::vector;

constexpr int kModels = 1;
const std::string feature_names[] = {
    "Constant",
    "Cast",
    "Variable",
    "Param",
    "Add",
    "Sub",
    "Mod",
    "Mul",
    "Div",
    "Min",
    "Max",
    "EQ",
    "NE",
    "LT",
    "LE",
    "And",
    "Or",
    "Not",
    "Select",
    "ImageCall",
    "FuncCall",
    "SelfCall",
    "ExternCall",
    "Let",
    "Pointwise other func",
    "Pointwise self-call",
    "Pointwise input image access",
    "Pointwise stores",
    "Transpose other func",
    "Transpose self-call",
    "Transpose input image access",
    "Transpose stores",
    "Broadcast other func",
    "Broadcast self-call",
    "Broadcast input image access",
    "Broadcast stores",
    "Slice other func",
    "Slice self-call",
    "Slice input image access",
    "Slice stores",
};
const std::string type_names[] = {"Bool", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"};

struct Sample {
    vector<float> runtimes;  // in msec
    double prediction[kModels];
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


// Load all the samples, reading filenames from stdin
map<int, PipelineSample> load_samples() {
    map<int, PipelineSample> result;
    vector<float> scratch(10 * 1024 * 1024);

    int best = -1;
    float best_runtime = 1e20f;
    string best_path;

    size_t num_read = 0, num_unique = 0;
    while (!std::cin.eof()) {
        string s;
        std::cin >> s;
        if (s.empty()) {
            continue;
        }
        if (!ends_with(s, ".sample")) {
            std::cout << "Skipping file: " << s << "\n";
            continue;
        }
        std::ifstream file(s);
        file.read((char *)(scratch.data()), scratch.size() * sizeof(float));
        const size_t floats_read = file.gcount() / sizeof(float);
        const size_t num_features = floats_read - 3;
        const size_t features_per_stage = head2_w + (head1_w + 1) * head1_h;
        file.close();
        // Note we do not check file.fail(). The various failure cases
        // are handled below by checking the number of floats read. We
        // expect truncated files if the benchmarking or
        // autoscheduling procedure crashes and want to filter them
        // out with a warning.

        if (floats_read == scratch.size()) {
            std::cout << "Too-large sample: " << s << " " << floats_read << "\n";
            continue;
        }
        if (num_features % features_per_stage != 0) {
            std::cout << "Truncated sample: " << s << " " << floats_read << "\n";
            continue;
        }
        const size_t num_stages = num_features / features_per_stage;

        const float runtime = scratch[num_features];
        if (runtime > 100000 || runtime < 0.1) {  // Don't try to predict runtime over 100s
            std::cout << "Implausible runtime in ms: " << runtime << "\n";
            continue;
        }

        int pipeline_id = *((int32_t *)(&scratch[num_features + 1]));
        const int schedule_id = *((int32_t *)(&scratch[num_features + 2]));

        if (runtime < best_runtime) {
            best_runtime = runtime;
            best = schedule_id;
            best_path = s;
        }

        uint64_t pipeline_hash = 0;
        for (size_t i = 0; i < num_stages; i++) {
            pipeline_hash =
                hash_floats(pipeline_hash,
                            &scratch[i * features_per_stage + head2_w],
                            &scratch[(i + 1) * features_per_stage]);
        }

        // Just use the hash as the id. Hash collisions are very very unlikely.
        PipelineSample &ps = result[pipeline_hash];

        if (ps.pipeline_features.data() == nullptr) {
            ps.pipeline_id = pipeline_id;
            ps.num_stages = (int)num_stages;
            ps.pipeline_features = Buffer<float>(head1_w, head1_h, num_stages);
            ps.fastest_runtime = 1e30f;
            for (size_t i = 0; i < num_stages; i++) {
                for (int x = 0; x < head1_w; x++) {
                    for (int y = 0; y < head1_h; y++) {
                        float f = scratch[i * features_per_stage + (x + 1) * 7 + y + head2_w];
                        if (f < 0 || std::isnan(f)) {
                            std::cout << "Negative or NaN pipeline feature: " << x << " " << y << " " << i << " " << f << "\n";
                        }
                        ps.pipeline_features(x, y, i) = f;
                    }
                }
            }

            ps.pipeline_hash = pipeline_hash;
        } else {
            // Even for a huge number of pipelines, a hash collision is
            // vanishingly unlikely. Still, this will detect ones that are going
            // to cause UB during training:
            if ((int)num_stages != ps.num_stages) {
                std::cout << "Hash collision: two pipelines with a different number of stages both hashed to " << pipeline_hash << "\n";
                continue;
            }
        }

        uint64_t schedule_hash = 0;
        for (size_t i = 0; i < num_stages; i++) {
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
                it->second.filename = s;
            } else {
                it->second.runtimes.push_back(runtime);
            }
            if (runtime < ps.fastest_runtime) {
                ps.fastest_runtime = runtime;
                ps.fastest_schedule_hash = schedule_hash;
            }
        } else {
            Sample sample;
            sample.filename = s;
            sample.runtimes.push_back(runtime);
            for (double &d : sample.prediction) {
                d = 0.0;
            }
            sample.schedule_id = schedule_id;
            sample.schedule_features = Buffer<float>(head2_w, num_stages);

            bool ok = true;
            for (size_t i = 0; i < num_stages; i++) {
                for (int x = 0; x < head2_w; x++) {
                    float f = scratch[i * features_per_stage + x];
                    if (f < 0 || f > 1e14 || std::isnan(f)) {
                        std::cout << "Negative or implausibly large schedule feature: " << i << " " << x << " " << f << "\n";
                        // Something must have overflowed
                        ok = false;
                    }
                    sample.schedule_features(x, i) = f;
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
            // std::cout << "Unique sample: " << leaf(p.second.filename) << " : " << p.second.runtimes[0] << "\n";
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
            // double stddev = std::sqrt(variance_sum / count);
            // std::cout << "Noise level: " << stddev << "\n";
        }
    }

    std::cout << "Distinct pipelines: " << result.size() << "\n";

    std::ostringstream o;
    o << "Best runtime is " << best_runtime << " msec, from schedule id " << best << " in file " << best_path << "\n";
    std::cout << o.str();
    return result;
}

}  // namespace

float compute_variance(const std::vector<float>& numbers) {
    float sum = 0.0f;
    int count = numbers.size();
    float sum_squares = 0.0f;
    for (size_t i = 0; i < numbers.size(); ++i) {
        sum += numbers[i];
    }
    float mean_value = (sum / count);
    for (size_t i = 0; i < numbers.size(); ++i) {
        sum_squares += ((numbers[i] - mean_value) * (numbers[i] - mean_value));
    }
    float variance = sum_squares / (count - 1);
    return variance;
}

int main(int argc, char **argv) {
    auto samples = load_samples();
    // 40 * 7 -> 280
    std::vector<std::vector<float>> pipeline_features_accum(head1_w * head1_h);
    for (const auto& pipe: samples) {
        size_t num_stages = pipe.second.num_stages;
        for (size_t i = 0; i < num_stages; i++) {
            for (int x = 0; x < head1_w; x++) {
                for (int y = 0; y < head1_h; y++) {
                    pipeline_features_accum[x * head1_h + y].push_back(pipe.second.pipeline_features(x, y, i));
                }
            }
        }
    }
    for (int y = 0; y < head1_h; y++) {
        std::cout << "Type " << type_names[y] << ":\n";
        for (int x = 0; x < head1_w; x++) {
            float variance = compute_variance(pipeline_features_accum[x * head1_h + y]);
            std::cout << "Variance of " << feature_names[x] << ": " << variance << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}