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
    return serialized.size();
}
