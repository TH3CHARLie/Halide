
#include "Halide.h"

using namespace Halide;


bool is_vectorizable(const std::string &var, const Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Split> &splits = schedule.splits();
    if (splits.size() == 0) {
        return false;
    }
    std::vector<std::string> const_factor_vars;
    for (int idx = 0; idx < splits.size(); ++idx) {
        auto &split = splits[idx];
        if (split.split_type == Internal::Split::SplitType::SplitVar) {
            if (std::find(const_factor_vars.begin(), const_factor_vars.end(), split.old_var) != const_factor_vars.end()) {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var || split.outer == var) {
                        return true;
                    } else {
                        const_factor_vars.push_back(split.inner);
                        const_factor_vars.push_back(split.outer);
                    }
                }
            } else {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var) {
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
    return false;
}


int main(int argc, char **argv) {
    const int width = 128;
    const int height = 128;
    Func input("input");
    Func local_sum("local_sum");
    Func blurry("blurry");
    Var x("x"), y("y"), yi("yi"), yo("yo"), xi("xi"), xo("xo"), yio("yio"), yii("yii"), xio("xio"), xii("xii"), yofxi("yofxi"), haha("haha"), xfy("xfy");
    input(x, y) = 2 * x + 5 * y;
    RDom r(-2, 5, -2, 5);
    local_sum(x, y) = 0;
    local_sum(x, y) += input(x + r.x, y + r.y);
    blurry(x, y) = cast<int32_t>(local_sum(x, y) / 25);
    Pipeline p({blurry});
    local_sum.split(y, yi, yo, 8).split(x, xi, xo, 16).reorder(xo, yi, yo, xi).parallel(yi).fuse(yo, xi, yofxi).vectorize(yofxi);
    local_sum.update(0).fuse(x, y, xfy);
    // local_sum.update(0).split(x, xo, xi, 16, TailStrategy::ShiftInwards);
    // blurry.rename(x, haha);
    // blurry.split(x, xo, xi, 16, TailStrategy::Auto);
    blurry.split(x, xo, xi, 16).split(y, yo, yi, 32).fuse(xi, yo, xfy);
    blurry.vectorize(xfy);
    std::cout << std::boolalpha <<  is_vectorizable("x.xi.xfy", blurry.function(), blurry.function().definition(), 0) << std::endl;
    {
        auto bounds = local_sum.function().schedule().bounds();
        std::cout << "bounds size :" << bounds.size() << std::endl;
        for (const auto &b : bounds) {
            std::cout << b.var << " " << b.extent << std::endl;
        }
        auto schedule = blurry.function().definition().schedule();
        std::vector<Internal::Split> splits = schedule.splits();
        std::cout << "splits size :" << splits.size() << std::endl;
        // remove this split from the splits
        splits.erase(splits.begin());
        std::vector<Internal::Dim> dims = schedule.dims();
        std::cout << "dim list" << std::endl;
        for (const auto &d : dims) {
            std::cout << d.var << std::endl;
        }
    }
    p.compile_to_module({}, "blurry", Halide::Target("host"));
    // serialize_pipeline(p, "blurry_wrong.hlpipe");
    return 0;
}
