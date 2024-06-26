#include "PrintLoopNest.h"
#include "AllocationBoundsInference.h"
#include "Bounds.h"
#include "BoundsInference.h"
#include "FindCalls.h"
#include "Func.h"
#include "Function.h"
#include "IRPrinter.h"
#include "RealizationOrder.h"
#include "RemoveExternLoops.h"
#include "RemoveUndef.h"
#include "ScheduleFunctions.h"
#include "Simplify.h"
#include "SimplifyCorrelatedDifferences.h"
#include "SimplifySpecializations.h"
#include "SlidingWindow.h"
#include "Target.h"
#include "UniquifyVariableNames.h"
#include "WrapCalls.h"

#include <tuple>

namespace Halide {
namespace Internal {

using std::map;
using std::string;
using std::vector;

namespace {

class PrintLoopNest : public IRVisitor {
public:
    PrintLoopNest(std::ostream &output, const map<string, Function> &e)
        : out(output), env(e) {
    }

private:
    std::ostream &out;
    const map<string, Function> &env;
    int indent = 0;

    Scope<Expr> constants;

    using IRVisitor::visit;

    Indentation get_indent() const {
        return Indentation{indent};
    }

    string simplify_var_name(const string &s) {
        return simplify_name(s, false);
    }

    string simplify_func_name(const string &s) {
        return simplify_name(s, true);
    }

    string simplify_name(const string &s, bool is_func) {
        // Trim the function name and stage number from the for loop,
        // as well as any uniqueness $n suffixes on variables.
        std::ostringstream trimmed_name;

        bool keep = is_func;
        int dot_count = 0;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '.') {
                dot_count++;
                if (dot_count >= 2) {
                    if (dot_count == 2) {
                        i++;
                    }
                    keep = true;
                }
            }
            if (s[i] == '$') {
                keep = false;
            }
            if (keep) {
                trimmed_name << s[i];
            }
        }

        return trimmed_name.str();
    }

    void visit(const For *op) override {
        string simplified_loop_var_name = simplify_var_name(op->name);

        out << get_indent() << op->for_type << " " << simplified_loop_var_name;

        // If the min or extent are constants, print them. At this
        // stage they're all variables.
        Expr min_val = op->min, extent_val = op->extent;
        const Variable *min_var = min_val.as<Variable>();
        const Variable *extent_var = extent_val.as<Variable>();
        if (min_var) {
            if (const Expr *e = constants.find(min_var->name)) {
                min_val = *e;
            }
        }

        if (extent_var) {
            if (const Expr *e = constants.find(extent_var->name)) {
                extent_val = *e;
            }
        }

        if (extent_val.defined() && is_const(extent_val) &&
            min_val.defined() && is_const(min_val)) {
            Expr max_val = simplify(min_val + extent_val - 1);
            out << " in [" << min_val << ", " << max_val << "]";
        }

        out << op->device_api;

        out << ":\n";
        indent += 2;
        op->body.accept(this);
        indent -= 2;
    }

    void visit(const Realize *op) override {
        // If the storage and compute levels for this function are
        // distinct, print the store level too.
        auto it = env.find(op->name);
        if (it != env.end() &&
            !(it->second.schedule().store_level() ==
              it->second.schedule().compute_level())) {
            out << get_indent();
            out << "store " << simplify_func_name(op->name) << ":\n";
            indent += 2;
            op->body.accept(this);
            indent -= 2;
        } else {
            op->body.accept(this);
        }
    }

    void visit(const ProducerConsumer *op) override {
        out << get_indent();
        if (op->is_producer) {
            out << "produce " << simplify_func_name(op->name) << ":\n";
        } else {
            out << "consume " << simplify_func_name(op->name) << ":\n";
        }
        indent += 2;
        op->body.accept(this);
        indent -= 2;
    }

    void visit(const Provide *op) override {
        out << get_indent() << simplify_func_name(op->name) << "(...) = ...\n";
    }

    void visit(const LetStmt *op) override {
        if (is_const(op->value)) {
            ScopedBinding<Expr> bind(constants, op->name, op->value);
            op->body.accept(this);
        } else {
            op->body.accept(this);
        }
    }
};

}  // namespace

string print_loop_nest(const vector<Function> &output_funcs) {
    // Do the first part of lowering:

    // Create a deep-copy of the entire graph of Funcs.
    auto [outputs, env] = deep_copy(output_funcs, build_environment(output_funcs));

    // Output functions should all be computed and stored at root.
    for (const Function &f : outputs) {
        Func(f).compute_root().store_root();
    }

    // Finalize all the LoopLevels
    for (auto &iter : env) {
        iter.second.lock_loop_levels();
    }

    // Substitute in wrapper Funcs
    env = wrap_func_calls(env);

    // Compute a realization order and determine group of functions which loops
    // are to be fused together
    auto [order, fused_groups] = realization_order(outputs, env);

    // Try to simplify the RHS/LHS of a function definition by propagating its
    // specializations' conditions
    simplify_specializations(env);

    // For the purposes of printing the loop nest, we don't want to
    // worry about which features are and aren't enabled.
    Target target = get_host_target();
    for (DeviceAPI api : all_device_apis) {
        target.set_feature(target_feature_for_device_api(DeviceAPI(api)));
    }

    bool any_memoized = false;
    // Schedule the functions.
    Stmt s = schedule_functions(outputs, fused_groups, env, target, any_memoized);

    // Compute the maximum and minimum possible value of each
    // function. Used in later bounds inference passes.
    FuncValueBounds func_bounds = compute_function_value_bounds(order, env);

    // This pass injects nested definitions of variable names, so we
    // can't simplify statements from here until we fix them up. (We
    // can still simplify Exprs).
    s = bounds_inference(s, outputs, order, fused_groups, env, func_bounds, target);
    s = remove_extern_loops(s);
    s = sliding_window(s, env);
    s = simplify_correlated_differences(s);
    s = allocation_bounds_inference(s, env, func_bounds);
    s = remove_undef(s);
    s = uniquify_variable_names(s);
    s = simplify(s, false);

    // Now convert that to pseudocode
    std::ostringstream sstr;
    PrintLoopNest pln(sstr, env);
    s.accept(&pln);
    return sstr.str();
}

}  // namespace Internal
}  // namespace Halide
