
#include "Halide.h"
#include <random>
using namespace Halide;

std::vector<VarOrRVar> get_stage_vars_or_rvars(const Stage &stage) {
    std::vector<VarOrRVar> vars;
    const auto &dims = stage.get_schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            if (d.is_rvar()) {
                vars.push_back(RVar(d.var));
            } else {
                vars.push_back(Var(d.var));
            }
        }
    }
    return vars;
}

bool is_vectorizable(const std::string &var, Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Split> &splits = schedule.splits();
    if (splits.size() == 0) {
        std::cout << "here we first return false\n";
        return false;
    }
    std::vector<std::string> const_factor_vars;
    for (int idx = 0; idx < splits.size(); ++idx) {
        auto &split = splits[idx];
        if (split.split_type == Internal::Split::SplitType::SplitVar) {
            if (std::find(const_factor_vars.begin(), const_factor_vars.end(), split.old_var) != const_factor_vars.end()) {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var || split.outer == var) {
                        std::cout << "here we return true\n";
                        return true;
                    } else {
                        const_factor_vars.push_back(split.inner);
                        const_factor_vars.push_back(split.outer);
                    }
                }
            } else {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var) {
                        std::cout << "here we return true1 2\n";
                        return true;
                    } else {
                        const_factor_vars.push_back(split.inner);
                    }
                }
            }
        } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
            if (split.old_var == var) {
                return is_vectorizable(split.inner, function, definition, stage_idx) && is_vectorizable(split.outer, function, definition, stage_idx);
            }
        }
    }
    std::cout << "here we return fals 222e\n";
    return false;
}

Internal::ForType random_for_type(std::mt19937 &rng) {
    std::uniform_int_distribution<int> for_type_dist(0, 3);
    int for_type_idx = for_type_dist(rng);
    return static_cast<Internal::ForType>(for_type_idx);
}



void mutate_change_existing_dim(Internal::Function &function, Internal::Definition &definition, int stage_idx, int try_idx) {
    // bool is_inlined = false;
    // function.schedule().compute_level().lock();
    // is_inlined = function.schedule().compute_level().is_inlined();
    // function.schedule().compute_level().unlock();
    // // if the function is inlined, we don't mutate the dim
    // if (is_inlined) {
    //     return;
    // }


    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Dim> &dims = schedule.dims();
    // if (dims.size() == 0) {
    //     return;
    // }
    // std::uniform_int_distribution<int> dim_idx_dist(0, dims.size() - 1);
    std::mt19937 rng(try_idx);
    int dim_idx = 1;
    Internal::Dim &dim = dims[dim_idx];
    std::cout << "dim's name " << dim.var << std::endl;
    if (dim.var == Var::outermost().name()) {
        return;
    }
    // TODO: we want to suppress some Can only vectorize/unroll for loops over a constant extent errors
    Internal::ForType for_type = random_for_type(rng);
    std::cout << "for_type " << for_type << std::endl;
    if (for_type == Internal::ForType::Vectorized || for_type == Internal::ForType::Unrolled) {
        if (is_vectorizable(dim.var, function, definition, stage_idx)) {
            dim.for_type = for_type;
        }
    } else {
        dim.for_type = for_type;
    }
}

// void mutate_remove_split(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
//     Stage stage(function, definition, stage_idx);
//     std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
//     std::vector<Internal::Split> splits = definition.schedule().splits();
//     if (splits.size() == 0) {
//         return;
//     }
//     int split_idx = 3;
//     Internal::Split split = splits[split_idx];
//     std::cout << "split being removed " << split.old_var << " " << split.inner << " " << split.outer << " " << split.factor << "\n";
//     std::set<std::string> removed_vars;

//     auto add_to_removed_vars = [&removed_vars](const Internal::Split &split) {
//         if (split.split_type == Internal::Split::SplitType::SplitVar) {
//             removed_vars.insert(split.inner);
//             removed_vars.insert(split.outer);
//         } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
//             removed_vars.insert(split.old_var);
//         } else if (split.split_type == Internal::Split::SplitType::RenameVar) {
//             removed_vars.insert(split.outer);
//         }
//     };

//     auto need_remove = [&removed_vars](const Internal::Split &split) {
//         if (split.split_type == Internal::Split::SplitType::SplitVar) {
//             return removed_vars.count(split.old_var) != 0;
//         } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
//             return removed_vars.count(split.inner) != 0 || removed_vars.count(split.outer) != 0;
//         } else if (split.split_type == Internal::Split::SplitType::RenameVar) {
//             return removed_vars.count(split.old_var) != 0;
//         }
//         return false;
//     };

//     add_to_removed_vars(split);
//     std::vector<int> removed_split_idxs;
//     removed_split_idxs.push_back(split_idx);
//     for (int idx = split_idx + 1; idx < splits.size(); ++idx) {
//         if (need_remove(splits[idx])) {
//             add_to_removed_vars(splits[idx]);
//             removed_split_idxs.push_back(idx);
//         }
//     }
//     std::vector<int> removed_dim_idxs;
//     std::vector<Internal::Dim> dims = definition.schedule().dims();
//     for (int idx = 0; idx < dims.size(); ++idx) {
//         if (dims[idx].var != Var::outermost().name()) {
//             if (removed_vars.count(dims[idx].var) != 0) {
//                 removed_dim_idxs.push_back(idx);
//             }
//         }
//     }
//     std::vector<Internal::Split> new_splits;
//     definition.schedule().splits().clear();
//     definition.schedule().dims().clear();
//     definition.schedule().dims().push_back(Internal::Dim({Var::outermost().name(), ForType::Serial, DeviceAPI::None, DimType::PureVar}));
//     // for (int idx = 0; idx < splits.size(); ++idx) {
//     //     if (std::find(removed_split_idxs.begin(), removed_split_idxs.end(), idx) == removed_split_idxs.end()) {
//     //         new_splits.push_back(splits[idx]);
//     //     }
//     // }
//     // definition.schedule().splits() = new_splits;
//     // std::vector<Internal::Dim> new_dims;
//     // for (int idx = 0; idx < dims.size(); ++idx) {
//     //     if (std::find(removed_dim_idxs.begin(), removed_dim_idxs.end(), idx) == removed_dim_idxs.end()) {
//     //         new_dims.push_back(dims[idx]);
//     //     }
//     // }
//     // definition.schedule().dims() = new_dims;
//     for (int idx = 0; idx < new_splits.size(); ++idx) {
//         if (new_splits[idx].split_type == Internal::Split::SplitType::SplitVar) {
//             std::cout << "new split " << new_splits[idx].old_var << " " << new_splits[idx].inner << " " << new_splits[idx].outer << " " << new_splits[idx].factor << "\n";
//             stage.split(new_splits[idx].old_var, new_splits[idx].inner, new_splits[idx].outer, new_splits[idx].factor, new_splits[idx].tail);
//         } else if (new_splits[idx].split_type == Internal::Split::SplitType::FuseVars) {
//             std::cout << "new split fuse " << new_splits[idx].old_var << " " << new_splits[idx].inner << " " << new_splits[idx].outer << "\n";
//             stage.fuse(new_splits[idx].inner, new_splits[idx].outer, new_splits[idx].old_var);
//         } else if (new_splits[idx].split_type == Internal::Split::SplitType::RenameVar) {
//             std::cout << "new split rename " << new_splits[idx].old_var << " " << new_splits[idx].inner << " " << new_splits[idx].outer << "\n";
//         }
//     }
//     return;
// }

int main(int argc, char **argv) {
    ImageParam input(Float(32), 2, "input");
    const float r_sigma = 0.1;
    const int s_sigma = 8;
    Func bilateral_grid{"bilateral_grid"};

    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition
    Func clamped = Halide::BoundaryConditions::repeat_edge(input);

    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma / 2, y * s_sigma + r.y - s_sigma / 2);
    val = clamp(val, 0.0f, 1.0f);

    Expr zi = cast<int>(val * (1.0f / r_sigma) + 0.5f);

    Func histogram("histogram");
    histogram(x, y, z, c) = 0.0f;
    histogram(x, y, zi, c) += mux(c, {val, 1.0f});

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurz(x, y, z, c) = (histogram(x, y, z - 2, c) +
                            histogram(x, y, z - 1, c) * 4 +
                            histogram(x, y, z, c) * 6 +
                            histogram(x, y, z + 1, c) * 4 +
                            histogram(x, y, z + 2, c));
    blurx(x, y, z, c) = (blurz(x - 2, y, z, c) +
                            blurz(x - 1, y, z, c) * 4 +
                            blurz(x, y, z, c) * 6 +
                            blurz(x + 1, y, z, c) * 4 +
                            blurz(x + 2, y, z, c));
    blury(x, y, z, c) = (blurx(x, y - 2, z, c) +
                            blurx(x, y - 1, z, c) * 4 +
                            blurx(x, y, z, c) * 6 +
                            blurx(x, y + 1, z, c) * 4 +
                            blurx(x, y + 2, z, c));

    // Take trilinear samples to compute the output
    val = clamp(input(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f / r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x / s_sigma;
    Expr yi = y / s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y, c) =
        lerp(lerp(lerp(blury(xi, yi, zi, c), blury(xi + 1, yi, zi, c), xf),
                    lerp(blury(xi, yi + 1, zi, c), blury(xi + 1, yi + 1, zi, c), xf), yf),
                lerp(lerp(blury(xi, yi, zi + 1, c), blury(xi + 1, yi, zi + 1, c), xf),
                    lerp(blury(xi, yi + 1, zi + 1, c), blury(xi + 1, yi + 1, zi + 1, c), xf), yf),
                zf);

    // Normalize
    bilateral_grid(x, y) = interpolated(x, y, 0) / interpolated(x, y, 1);
    Pipeline p({bilateral_grid});
    // from apps/bilateral_grid/bilateral_grid_generator.cpp
    Var cfx("cfx"), co("co"), ci("ci"), zo("zo"), zi_("zi"), zio("zio"), zii("zii"), cio("cio"), cii("cii"), ciofx("ciofx"), ciofxo("ciofxo"), ciofxi("ciofxi"), ciofxorn("ciofxorn");
    blurz.compute_root()
        .reorder(c, z, x, y);
        // .vectorize(x, 8)
        // .unroll(c);
    histogram.compute_root().split(c, co, ci, 4).split(ci, cio, cii, 8).fuse(cio, x, ciofx).split(ciofx, ciofxo, ciofxi, 2).rename(ciofxo, ciofxorn); //.split(zi_, zio, zii, 1, TailStrategy::Predicate); //.split(c, co, ci, 5, TailStrategy::Predicate).reorder(x, co, z, ci, y);
    // split(c, co, ci, 5, TailStrategy::Predicate).reorder(x, co, z, ci, y).unroll(y)
    histogram.update()
        .reorder(c, r.x, r.y, x, y);
        // .unroll(c);
    blurx.compute_root()
        .reorder(c, x, z, y)
        .parallel(y);
        // .vectorize(x, 8)
        // .unroll(c);
    blury.compute_root()
        .reorder(c, x, y, z)
        .reorder_storage(c, z, x, y);
        // .vectorize(x, 8)
        // .unroll(c);
    bilateral_grid.compute_root()
        .parallel(y, 8);
//        .vectorize(x, 8);
    // input.set(input_buf);
    // p.compile_to_module({input}, "bilateral_grid", Halide::Target("host"));
    {
        std::vector<Internal::Function> outputs;
        std::set<std::string> outputs_names;
        for (const auto &f : p.outputs()) {
            outputs.push_back(f.function());
            outputs_names.insert(f.name());
        }
        std::map<std::string, Internal::Function> env = build_environment(outputs);
        std::vector<std::string> order = topological_order(outputs, env);
        for (int i = 0; i < order.size(); ++i) {
            std::string func_name = order[i];
            Internal::Function func = env[func_name];
            if (func_name == "histogram") {
                int stage_idx = 0;
                Internal::Definition definition = func.definition();
                Internal::StageSchedule &schedule = definition.schedule();
                Stage stage(func, definition, stage_idx);
                std::cout << "splits before removing\n";
                for (const auto &split : schedule.splits()) {
                    std::cout << split.old_var << " " << split.inner << " " << split.outer << " " << split.factor << " " << split.is_rename() << " " << split.is_fuse() << " " << split.is_split() << std::endl;
                }
                std::cout << "dims before removing\n";
                for (const auto &dim : schedule.dims()) {
                    std::cout << dim.var << " " << dim.for_type << std::endl;
                }
                // mutate_remove_split(func, definition, stage_idx);
                std::cout << "splits after removing\n";
                for (const auto &split : schedule.splits()) {
                    std::cout << split.old_var << " " << split.inner << " " << split.outer << " " << split.factor << " " << split.is_rename() << " " << split.is_fuse() << " " << split.is_split() << std::endl;
                }
                std::cout << "dims after removing\n";
                for (const auto &dim : schedule.dims()) {
                    std::cout << dim.var << " " << dim.for_type << std::endl;
                }
            }
        }
    }
    // serialize_pipeline(p, "bilateral_grid_no_unroll_vectorize.hlpipe");
    return 0;
}
