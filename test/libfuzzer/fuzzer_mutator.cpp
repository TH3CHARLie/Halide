#include "Halide.h"
#include <cstdio>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <exception>
#include <fstream>
#include <cassert>

using namespace Halide;

static int counter = 0;
static int internal_error_counter = 0;

std::string tail_strategy_code(TailStrategy strategy) {
    switch (strategy) {
        case TailStrategy::RoundUp:
            return "TailStrategy::RoundUp";
        case TailStrategy::GuardWithIf:
            return "TailStrategy::GuardWithIf";
        case TailStrategy::Predicate:
            return "TailStrategy::Predicate";
        case TailStrategy::PredicateLoads:
            return "TailStrategy::PredicateLoads";
        case TailStrategy::PredicateStores:
            return "TailStrategy::PredicateStores";
        case TailStrategy::ShiftInwards:
            return "TailStrategy::ShiftInwards";
        case TailStrategy::ShiftInwardsAndBlend:
            return "TailStrategy::ShiftInwardsAndBlend";
        case TailStrategy::RoundUpAndBlend:
            return "TailStrategy::RoundUpAndBlend";
        case TailStrategy::Auto:
            return "TailStrategy::Auto";
    }
    return "Unknown";
}

std::string stage_schedule_code(const Internal::StageSchedule &stage_schedule, std::string func_name) {
    std::ostringstream ss;
    std::vector<Internal::Split> splits = stage_schedule.splits();
    if (splits.size() == 0) {
        return "";
    }
    ss << func_name << ".";
    for (int split_idx = 0; split_idx < splits.size(); ++split_idx) {
        auto split_type = splits[split_idx].split_type;
        if (split_type == Internal::Split::SplitType::SplitVar) {
            ss << "split(" << splits[split_idx].old_var << ", " << splits[split_idx].outer << ", " << splits[split_idx].inner << ", " << splits[split_idx].factor << ", " << tail_strategy_code(splits[split_idx].tail) << ")";
        }
        else if (split_type == Internal::Split::SplitType::RenameVar) {
            ss << "rename(" << splits[split_idx].old_var << ", " << splits[split_idx].outer << ")";
        } else if (split_type == Internal::Split::SplitType::FuseVars) {
            ss << "fuse(" << splits[split_idx].inner << ", " << splits[split_idx].outer << ", " << splits[split_idx].old_var << ")";
        }
        if (split_idx != splits.size() - 1) {
            ss << ".";
        }
    }
    // output a reorder schedule even if the dims are not reordered
    std::vector<Internal::Dim> dims = stage_schedule.dims();
    if (dims.size() == 0) {
        return "";
    }
    ss << ".reorder(";
    for (int dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
        if (dims[dim_idx].var != "__outermost") {
            ss << dims[dim_idx].var;
            if (dim_idx != dims.size() - 2) {
                ss << ", ";
            }
        }
    }
    ss << ")";
    // output any parallel or vectorize dim
    for (int dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
        if (dims[dim_idx].for_type == Internal::ForType::Parallel) {
            ss << ".parallel(" << dims[dim_idx].var << ")";
        } else if (dims[dim_idx].for_type == Internal::ForType::Vectorized) {
            ss << ".vectorize(" << dims[dim_idx].var << ")";
        }
    }
    return ss.str();
}

std::string schedule_code(const Pipeline &p) {
    std::ostringstream ss;
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    // print loop schedule
    for (int i = 0; i < order.size(); ++i) {
        std::string func_name = order[i];
        Internal::Function func = env[func_name];
        for (int stage_idx = 0; stage_idx <= (int)func.updates().size(); ++stage_idx) {
            if (stage_idx == 0) {
                ss << stage_schedule_code(func.definition().schedule(), func_name);
            } else {
                std::string update_name = func_name + ".update(" + std::to_string(stage_idx - 1) + ")";
                ss << stage_schedule_code(func.update(stage_idx - 1).schedule(), update_name);
            }
            ss << "\n";
        }
        ss << "\n";
    }
    return ss.str();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    std::string output_dir = Internal::get_env_variable("FUZZER_OUTPUT_DIR");
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    std::ofstream out(output_dir + "/blurry_" + std::to_string(counter) + ".log");
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        deserialized = deserialize_pipeline(data_copy, params);
        deserialized.compile_to_module({}, "blurry", get_host_target());
        std::string code = schedule_code(deserialized);
        out << code;
        std::string hlpipe_file = output_dir + "/blurry_" + std::to_string(counter) + ".hlpipe";
        serialize_pipeline(deserialized, hlpipe_file);
    } catch (const CompileError &e) {
        out << "\nUser Error: " << e.what() << "\n";
    } catch (const InternalError &e) {
        out << "\nInternal Error: " << e.what() << "\n";
        internal_error_counter++;
    }
    counter++;
    if (internal_error_counter > 0) {
        std::cout << "current fuzzing counter " << counter << " internal error counter " << internal_error_counter << "\n";
    }
    return 0;
}

extern "C" size_t
LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

static int mutate_counter = 0;
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
    // Mutate data in place. Return the new size.
    std::cout << "size " << size << " max_size " << max_size << "\n";
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        deserialized = deserialize_pipeline(data_copy, params);
    } catch (const CompileError &e) {
        // user error means invalid input, so we just return 0
        return 0;
    } catch (const InternalError &e) {
        // we really shouldn't hitting any internal errors here, but if we do, we just return 0
        std::cerr << "InternalError during deserialization: " << e.what() << std::endl;
        return 0;
    }
    // mutate the deserialized pipeline
    ScheduleMutator mutator(deserialized, seed);
    Pipeline mutated = mutator.mutate();
    std::vector<uint8_t> serialized;
    serialize_pipeline(mutated, serialized);
    std::cout << "serialized size " << serialized.size() << "\n";
    // serialize the mutated pipeline
    mutate_counter++;
    std::memcpy(data, serialized.data(), serialized.size());
    return serialized.size();
}
