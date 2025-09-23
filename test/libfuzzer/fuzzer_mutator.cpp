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

std::string get_original_var_name(const std::string &var_name) {
    if (var_name.rfind('.' != std::string::npos)) {
        return var_name.substr(var_name.rfind('.') + 1, var_name.size() - 1 - var_name.rfind('.'));
    }
    return var_name;
}

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
    ss << func_name;
    if (splits.size() == 0) {
        ss << "";
    } else {
        ss << ".";
        for (int split_idx = 0; split_idx < splits.size(); ++split_idx) {
            auto split_type = splits[split_idx].split_type;
            std::string old_var = get_original_var_name(splits[split_idx].old_var);
            std::string outer = get_original_var_name(splits[split_idx].outer);
            std::string inner = get_original_var_name(splits[split_idx].inner);
            if (split_type == Internal::Split::SplitType::SplitVar) {
                ss << "split(" << old_var << ", " << outer << ", " << inner << ", " << splits[split_idx].factor << ", " << tail_strategy_code(splits[split_idx].tail) << ")";
            }
            else if (split_type == Internal::Split::SplitType::RenameVar) {
                ss << "rename(" << old_var << ", " << outer << ")";
            } else if (split_type == Internal::Split::SplitType::FuseVars) {
                ss << "fuse(" << inner << ", " << outer << ", " << old_var << ")";
            }
            if (split_idx != splits.size() - 1) {
                ss << ".";
            }
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
            ss << get_original_var_name(dims[dim_idx].var);
            if (dim_idx != dims.size() - 2) {
                ss << ", ";
            }
        }
    }
    ss << ")";
    // output any parallel or vectorize dim
    for (int dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
        if (dims[dim_idx].for_type == Internal::ForType::Parallel) {
            ss << ".parallel(" << get_original_var_name(dims[dim_idx].var) << ")";
        } else if (dims[dim_idx].for_type == Internal::ForType::Vectorized) {
            ss << ".vectorize(" << get_original_var_name(dims[dim_idx].var) << ")";
        } else if (dims[dim_idx].for_type == Internal::ForType::Unrolled) {
            ss << ".unroll(" << get_original_var_name(dims[dim_idx].var) << ")";
        }
    }
    return ss.str();
}

std::string print_pipeline_schedule(const Pipeline &p) {
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
                if (func.schedule().compute_level().lock().is_inlined()) {
                    ss << ".compute_inline()";
                } else if (func.schedule().compute_level().lock().is_root()) {
                    ss << ".compute_root()";
                }
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

static int counter = 0;
static int internal_error_counter = 0;
static int unknown_error_counter = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    std::string output_dir = Internal::get_env_variable("FUZZER_OUTPUT_DIR");
    std::string pipeline_name = Internal::get_env_variable("FUZZER_PIPELINE_NAME");
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    std::ofstream out(output_dir + "/blurry_" + std::to_string(counter) + ".log");
    if (!out.is_open()) {
        std::cerr << "failed to open log file\n";
        return 0;
    }
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        deserialized = deserialize_pipeline(data_copy, params);
        if (!deserialized.defined()) {
            std::cerr << "skip undefined pipeline\n";
            return 0;
        }
        std::string hlpipe_file = output_dir + "/bilateral_grid_" + std::to_string(counter) + ".hlpipe";
        serialize_pipeline(deserialized, hlpipe_file);
        std::string code = print_pipeline_schedule(deserialized);
        out << code;
        out.flush();
        ImageParam input(Float(32), 2, "input");
        deserialized.compile_to_module({input}, "bilateral_grid", get_host_target());
    } catch (const CompileError &e) {
        out << "\nUser Error: " << e.what() << "\n";
    } catch (const InternalError &e) {
        out << "\nInternal Error: " << e.what() << "\n";
        internal_error_counter++;
    } catch (const std::exception &e) {
        out << "\nUnknown Error: " << e.what() << "\n";
        unknown_error_counter++;
    }
    counter++;
    if (unknown_error_counter > 0 || internal_error_counter > 0) {
        std::cout << "current fuzzing counter " << counter << " internal error counter " << internal_error_counter << " unknown error counter " << unknown_error_counter << "\n";
    }
    return 0;
}

extern "C" size_t
LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

static int mutate_counter = 0;
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
    // Mutate data in place. Return the new size.
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        deserialized = deserialize_pipeline(data_copy, params);
        // mutate the deserialized pipeline
        ScheduleMutator mutator(deserialized, seed);
        Pipeline mutated = mutator.mutate();
        std::vector<uint8_t> serialized;
        serialize_pipeline(mutated, serialized);
        // serialize the mutated pipeline
        mutate_counter++;
        std::memcpy(data, serialized.data(), serialized.size());
        return serialized.size();
    } catch (const CompileError &e) {
        // user error means we mutate into some invalid inputs, so we just abort
        return 0;
    } catch (const InternalError &e) {
        // we really shouldn't hitting any internal errors here, but if we do, we just return 0
        std::cerr << "InternalError during mutation: " << e.what() << std::endl;
        return 0;
    }
}
