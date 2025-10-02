#include <Halide.h>
#include <iostream>

using namespace std;
using namespace Halide;
using namespace Halide::Internal;

void print_expr(const Expr &e) {
    std::cout << e << "\n";
}

void print_function_def(const Function &func) {
    // Get function args for LHS
    std::vector<std::string> args;
    for (const auto &arg : func.args()) {
        args.push_back(arg);  // func.args() returns vector<string>
    }
    std::string args_str = "(";
    for (size_t i = 0; i < args.size(); i++) {
        args_str += args[i];
        if (i < args.size() - 1) args_str += ", ";
    }
    args_str += ")";
    
    const Definition &init_def = func.definition();
    if (init_def.values().size() > 1) {
        for (size_t i = 0; i < init_def.values().size(); i++) {
            std::cout << func.name() << args_str << "[" << i << "] = ";
            print_expr(init_def.values()[i]);
        }
    } else {
        std::cout << func.name() << args_str << " = ";
        print_expr(init_def.values()[0]);
    }

    const vector<Definition> &updates = func.updates();
    for (size_t i = 0; i < updates.size(); i++) {
        const Definition &update = updates[i];
        
        // Print reduction domain if it exists
        if (update.schedule().rvars().size() > 0) {
            std::cout << "  where rvars: ";
            for (const auto &rvar : update.schedule().rvars()) {
                std::cout << rvar.var << " ";
            }
            std::cout << "\n";
        }
        
        // Get update args for LHS - use the same args as the pure definition
        std::vector<std::string> update_args = func.args();
        std::string update_args_str = "(";
        for (size_t i = 0; i < update_args.size(); i++) {
            update_args_str += update_args[i];
            if (i < update_args.size() - 1) update_args_str += ", ";
        }
        update_args_str += ")";
        
        if (update.values().size() > 1) {
            for (size_t j = 0; j < update.values().size(); j++) {
                std::cout << func.name() << update_args_str << "[" << j << "] = ";
                print_expr(update.values()[j]);
            }
        } else {
            std::cout << func.name() << update_args_str << " = ";
            print_expr(update.values()[0]);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pipeline.hlpipe>\n";
        return 1;
    }

    const char* pipeline_path = argv[1];
    
    // Deserialize the pipeline
    std::map<std::string, Parameter> params = deserialize_parameters(pipeline_path);
    Pipeline pipeline = deserialize_pipeline(pipeline_path, params);
    std::cout << "Pipeline deserialized\n";

    std::vector<Internal::Function> outputs;
    std::set<std::string> output_function_names;
    for (const auto &f : pipeline.outputs()) {
        outputs.push_back(f.function());
        output_function_names.insert(f.name());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    // Print functions in topological order
    std::cout << "Deserialized Pipeline code (without Schedule)\n";
    for (const auto &name : order) {
        const auto &func = env[name];
        print_function_def(func);
    }
    return 0;
}