#include "Halide.h"

using namespace Halide;
using namespace Halide::Internal;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pipeline.hlpipe>\n";
        return 1;
    }
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }

    const char* pipeline_path = argv[1];
    
    // Deserialize the pipeline
    std::map<std::string, Parameter> params = deserialize_parameters(pipeline_path);
    Pipeline pipeline = deserialize_pipeline(pipeline_path, params);
    std::cout << "Pipeline deserialized\n";

    // std::string schedule_str = print_pipeline_schedule(pipeline);
    // std::cout << "Pipeline schedule:\n";
    // std::cout << schedule_str << "\n";

    std::cout << "try compile it now...\n";
    std::vector<uint8_t> serialized;
    serialize_pipeline(pipeline, serialized);
    std::cout << "serialized size: " << serialized.size() << "\n";

    // pipeline.compile_to_module({}, "fuzz_module", get_host_target());
    // try {
        pipeline.compile_to_lowered_stmt("fuzz_pipeline.html", pipeline.infer_arguments(), HTML);
        // pipeline.compile_to_module({}, "fuzz_module", get_host_target());
    // } catch (const Halide::InternalError& e) {
    //     std::cerr << "Internal error during compilation: " << e.what() << "\n";
    //     return 1;
    // }
    // Clear all schedules
    // std::vector<Internal::Function> outputs;
    // std::set<std::string> output_function_names;
    // for (const auto &f : pipeline.outputs()) {
    //     outputs.push_back(f.function());
    //     output_function_names.insert(f.name());
    // }
    // std::map<std::string, Internal::Function> env = build_environment(outputs);
    // std::vector<std::string> order = topological_order(outputs, env);
    // clear_previous_schedule(pipeline, env, order);
    // std::cout << "All schedules cleared\n";
    // std::string cleared_schedule_str = print_pipeline_schedule(pipeline);
    // std::cout << "Cleared pipeline schedule:\n";
    // std::cout << cleared_schedule_str << "\n";
    return 0;
}