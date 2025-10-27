#include "Halide.h"
#include "fuzzer_utils.h"
#include <cstdio>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <exception>
#include <fstream>

using namespace Halide;



class BaselineScheduleGenerator {
public:
    BaselineScheduleGenerator(Pipeline p, unsigned int seed): pipeline(p), rng(seed) {}

    Pipeline generate();

private:
    Pipeline pipeline;

    std::mt19937 rng;

    // return a random integer in [min, max]
    int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng);
    }

    bool random_bool() {
        std::uniform_int_distribution<int> dist(0, 1);
        return dist(rng) == 1;
    }

    void generate_loop_schedule(Internal::Function &f);

    void generate_loop_schedule_per_stage(Internal::Function &f, Stage stage, int stage_index);

    void generate_compute_schedule(Internal::Function &f, const std::map<std::string, Internal::Function> &env, const std::vector<std::string> &order);

    void clear_previous_schedule(Pipeline &pipeline, std::map<std::string, Internal::Function> &env, const std::vector<std::string> &order) {
        for (int i = 0; i < order.size(); i++) {
            Internal::Function &f = env[order[i]];
            // reset function schedule
            std::cout << "before clearing schedule of function " << f.name() << "\n";
            std::cout << "f has " << f.schedule().bounds().size() << " bounds before clearing\n";
            std::cout << "f has " << f.schedule().storage_dims().size() << " storage dims before clearing\n";
            f.schedule() = Halide::Internal::FuncSchedule();
            // reset compute schedule to root (Note: inline is the default, but we choose to use compute_root instead)
            Func(f).compute_root();

            // 2. Reset the loop schedule for the pure definition.
            // Replacing the StageSchedule object is the cleanest way to clear
            // all splits, dims, reorders, etc., to their default state.
            f.definition().schedule() = Halide::Internal::StageSchedule();

            // 3. Reset the loop schedule for all update definitions.
            for (int u = 0; u < (int)f.updates().size(); u++) {
                f.update(u).schedule() = Halide::Internal::StageSchedule();
            }
        }    
    }

};

Pipeline BaselineScheduleGenerator::generate() {
    std::vector<Internal::Function> outputs;
    for (const auto &f : this->pipeline.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    clear_previous_schedule(this->pipeline, env, order);
    for (int i = 0; i < order.size(); i++) {
        generate_loop_schedule(env[order[i]]);
        generate_compute_schedule(env[order[i]], env, order);
    }
    return this->pipeline;
}


const int MAX_LOOP_SCHEDULE_ATTEMPTS = 8;

int split_factor_candidates[] = {1, 2, 4, 8};
TailStrategy tail_strategies_candidates[] = {TailStrategy::RoundUp, TailStrategy::GuardWithIf, TailStrategy::Predicate, /* temporarily disable TailStrategy::PredicateLoads, */ TailStrategy::PredicateStores, TailStrategy::ShiftInwards, TailStrategy::Auto};

void BaselineScheduleGenerator::generate_loop_schedule_per_stage(Internal::Function &f, Stage stage, int stage_index) {
    bool is_update = stage_index > 0;

    int num_attempts = random_int(1, MAX_LOOP_SCHEDULE_ATTEMPTS);
    for (int attempt = 0; attempt < num_attempts; attempt++) {
        // split, reorder, fuse, unroll, vectorize, parallel
        int random_loop_choice = random_int(0, 5);
        std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
        if (vars.size() == 0) {
            // no loops to schedule
            return;
        }
        switch (random_loop_choice) {
            case 0: {
                // split
                VarOrRVar var_picked = vars[random_int(0, vars.size() - 1)];
                int split_factor = split_factor_candidates[random_int(0, sizeof(split_factor_candidates) / sizeof(int) - 1)];
                TailStrategy tail_strategy = tail_strategies_candidates[random_int(0, sizeof(tail_strategies_candidates) / sizeof(TailStrategy) - 1)];
                std::string var_name = get_original_var_name(var_picked.name());
                if (var_picked.is_rvar) {
                    stage.split(RVar(var_name), RVar(var_name + "o"), RVar(var_name + "i"), split_factor, tail_strategy);
                } else {
                    stage.split(Var(var_name), Var(var_name + "o"), Var(var_name + "i"), split_factor, tail_strategy);
                }
                break;
            }
            case 1: {
                // reorder
                if (vars.size() < 2) {
                    continue;
                }
                
                // Create a new vector of VarOrRVars with unqualified names.
                std::vector<VarOrRVar> unqualified_vars;
                for (const auto& v : vars) {
                    std::string original_name = get_original_var_name(v.name());
                    unqualified_vars.emplace_back(original_name, v.is_rvar);
                }

                // Shuffle the new vector.
                std::shuffle(unqualified_vars.begin(), unqualified_vars.end(), rng);
                
                // Pass the clean, unqualified vars to the reorder call.
                stage.reorder(unqualified_vars);
                break;
            }
            case 2: {
                // fuse
                VarOrRVar var_picked = vars[random_int(0, vars.size() - 1)];
                // we need at least two vars to fuse
                if (vars.size() < 2) {
                    continue;
                }
                VarOrRVar inner_var = vars[random_int(0, vars.size() - 1)];
                VarOrRVar outer_var = vars[random_int(0, vars.size() - 1)];
                while (outer_var.name() == inner_var.name()) {
                    outer_var = vars[random_int(0, vars.size() - 1)];
                }
                std::string inner_name = get_original_var_name(inner_var.name());
                std::string outer_name = get_original_var_name(outer_var.name());
                // TODO: we rely on Halide to catch the error here, but we should do better
                if (random_bool()) {
                    std::string fused_var_name = "r_" + inner_name + "_" + outer_name + "_f";
                    stage.fuse(VarOrRVar(inner_name, inner_var.is_rvar), VarOrRVar(outer_name, outer_var.is_rvar) , RVar(fused_var_name));
                } else {
                    std::string fused_var_name = inner_name + "_" + outer_name + "_f";
                    stage.fuse(VarOrRVar(inner_name, inner_var.is_rvar), VarOrRVar(outer_name, outer_var.is_rvar) , Var(fused_var_name));
                }
                break;
            }
            case 3: {
                // unroll
                VarOrRVar var_picked = vars[random_int(0, vars.size() - 1)];
                std::string var_name = get_original_var_name(var_picked.name());
                if (var_picked.is_rvar) {
                    stage.unroll(RVar(var_name));
                } else {
                    stage.unroll(Var(var_name));
                }
                break;
            }
            case 4: {
                // vectorize
                VarOrRVar var_picked = vars[random_int(0, vars.size() - 1)];
                std::string var_name = get_original_var_name(var_picked.name());
                if (var_picked.is_rvar) {
                    stage.vectorize(RVar(var_name));
                } else {
                    stage.vectorize(Var(var_name));
                }
                break;
            }
            case 5: {
                // parallel
                VarOrRVar var_picked = vars[random_int(0, vars.size() - 1)];
                std::string var_name = get_original_var_name(var_picked.name());
                if (var_picked.is_rvar) {
                    stage.parallel(RVar(var_name));
                } else {
                    stage.parallel(Var(var_name));
                }
                break;
            }
            default:
                break;
        }
    }
}

void BaselineScheduleGenerator::generate_loop_schedule(Internal::Function &f) {
    for (int stage_idx = 0; stage_idx <= (int)f.updates().size(); stage_idx++) {
        if (stage_idx == 0) {
            // pure definition
            generate_loop_schedule_per_stage(f, Stage(f, f.definition(), 0), stage_idx);
        } else {
            // update definition
            generate_loop_schedule_per_stage(f, Stage(f, f.update(stage_idx - 1), stage_idx), stage_idx);
        }
    }
}

void BaselineScheduleGenerator::generate_compute_schedule(Internal::Function &f, const std::map<std::string, Internal::Function> &env, const std::vector<std::string> &order) {
    int random_compute_choice = random_int(0, 2);
    switch (random_compute_choice) {
        case 0: {
            Func(f).compute_root();
            break;
        }
        case 1: {
            // compute inline
            Func(f).compute_inline();
            break;
        }
        case 2: {
            // compute at
            // randomly picking a function and one of its loop variables
            if (order.empty()) {
                Halide::Func(f).compute_root();
                break;
            }

            int consumer_idx = random_int(0, order.size() - 1);
            const std::string& random_func_name = order[consumer_idx];
            Halide::Internal::Function random_func = env.at(random_func_name);

            std::vector<VarOrRVar> random_func_vars = get_function_vars(random_func);
            if (random_func_vars.empty()) {
                Halide::Func(f).compute_root();
                break;
            }
            VarOrRVar var_picked = random_func_vars[random_int(0, random_func_vars.size() - 1)];
            if (var_picked.is_rvar) {
                Func(f).compute_at(Func(random_func), RVar(var_picked.name()));
            } else {
                Func(f).compute_at(Func(random_func), Var(var_picked.name()));
            }
            break;
        }
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    // Bail on tiny inputs; avoid OOB in the deserializer.
    static constexpr size_t kMinSerializedSize = 16; // conservative
    if (size < kMinSerializedSize) return 0;

    // 1. Create a buffer from the fuzzer-provided data.
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        // 2. Deserialize the buffer into a Halide Pipeline.
        // This is the primary entry point for testing.
        deserialized = deserialize_pipeline(data_copy, params);
        if (!deserialized.defined()) {
            std::cerr << "skip undefined pipeline\n";
            return 0;
        }
        std::vector<uint8_t> serialized;
        serialize_pipeline(deserialized, serialized);
        std::cout << "serialized size: " << serialized.size() << "\n";
        // std::vector<Argument> args;

        // // 2. Populate it from the deserialized parameters.
        // //    The order often doesn't matter for compilation, but sorting by name
        // //    ensures deterministic behavior if it ever does.
        // for (const auto &pair : params) {
        //     args.push_back(pair.second.buffer());
        // }
        // TODO: we may need to add buffer parameters to the pipeline when compiling them
        // ImageParam input(Float(32), 2, "input");
        deserialized.compile_to_lowered_stmt("random_pipeline.html", {}, HTML);
        // deserialized.compile_to_module({}, "random_pipeline", get_host_target());
    } catch (const CompileError &e) {
        // This is an EXPECTED, user-facing error from an invalid schedule.
        // We catch it and return 0 to tell the fuzzer this is not a bug.
        return 0;
    }
    // --- Let all other exceptions crash! ---
    // If an InternalError or any other unexpected exception is thrown,
    // we do NOT catch it. The program will terminate, and libFuzzer
    // will correctly save the input as a crash.

    // If we reached this point, the input was valid and did not crash.
    return 0;
}

extern "C" size_t
LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
    // Mutate data in place. Return the new size.
    static constexpr size_t kMinSerializedSize = 16;

    // Early: tiny input? just delegate.
    if (size < kMinSerializedSize) {
        size_t out = LLVMFuzzerMutate(data, size, max_size);
        if (out == 0) { data[0] = 0; return 1; }   // <- never return 0
        return out;
    }
    std::vector<uint8_t> data_copy(data, data + size);
    std::map<std::string, Parameter> params;
    Pipeline deserialized;
    try {
        deserialized = deserialize_pipeline(data_copy, params);
        if (!deserialized.defined()) {
            return 0;
        }
        // generate a new schedule for this Pipeline
        BaselineScheduleGenerator generator(deserialized, seed);
        Pipeline new_generated_pipeline = generator.generate();
        std::vector<uint8_t> serialized;
        serialize_pipeline(new_generated_pipeline, serialized);
        if (serialized.size() == 0 || serialized.size() > max_size) {
            return 0;
        }
        // serialize the mutated pipeline
        std::memcpy(data, serialized.data(), serialized.size());
        return serialized.size();
    } catch (const CompileError &e) {
        // user error means we mutate into some invalid inputs, so we just abort
        size_t out = LLVMFuzzerMutate(data, size, max_size);
        if (out == 0) { data[0] = 0; return 1; }
        return out;
    }
    // as for InternalError, we shouldn't hitting any of them here, but if we do, we let it crash
    size_t out = LLVMFuzzerMutate(data, size, max_size);
    if (out == 0) { data[0] = 0; return 1; }
    return out;
}