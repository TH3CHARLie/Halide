#include "Halide.h"

namespace {

class SimplePipeline : public Halide::Generator<SimplePipeline> {
public:
    GeneratorParam<int> num_bins{"num_bins", 10, 1, 100};

    // GeneratorParam<int> stencils{"stencils", 32, 1, 100};

    Input<Buffer<uint16_t, 2>> input{"input"};
    Output<Buffer<uint16_t, 2>> output{"output"};

    void generate() {

        std::vector<Func> stages;

        Var x("x"), y("y");

        Func clamped("clamped");
        clamped = Halide::BoundaryConditions::repeat_edge(input);

        Func blurx("blurx");
        blurx(x, y) = (clamped(x - 1, y) +
                           2 * clamped(x, y) +
                           clamped(x + 1, y)) / 4;

        output(x, y) = blurx(x, y);

        /* ESTIMATES */
        // (This can be useful in conjunction with RunGen and benchmarks as well
        // as auto-schedule, so we do it in all cases.)
        {
            const int width = 1920;
            const int height = 1080;
            // Provide estimates on the input image
            input.set_estimates({{0, width}, {0, height}});
            // Provide estimates on the pipeline output
            output.set_estimates({{0, width}, {0, height}});
        }

        if (auto_schedule) {
            // nothing
        } else if (get_target().has_gpu_feature()) {
            clamped.compute_root();
            blurx.compute_root();
            output.compute_root();

        } else {
            clamped.compute_root();
            blurx.compute_root();
            output.compute_root();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(SimplePipeline, simple_pipeline)
