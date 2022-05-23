#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "simple_pipeline.h"
#ifndef NO_AUTO_SCHEDULE
#include "simple_pipeline_auto_schedule.h"
#endif

#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: ./filter input.png output.png timing_iterations\n"
               "e.g. ./filter input.png output.png 10\n");
        return 0;
    }

    int timing_iterations = atoi(argv[3]);

    Buffer<int16_t, 3> input_rgb = load_and_convert_image(argv[1]);

    Buffer<int16_t, 2> input = input_rgb.sliced(2, 0);

    Buffer<int16_t, 2> output(input.width(), input.height());

    simple_pipeline(input, output);

    // Timing code. Timing doesn't include copying the input data to
    // the gpu or copying the output back.

    // Manually-tuned version
    double min_t_manual = benchmark(timing_iterations, 10, [&]() {
        simple_pipeline(input, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

#ifndef NO_AUTO_SCHEDULE
    // Auto-scheduled version
    double min_t_auto = benchmark(timing_iterations, 10, [&]() {
        simple_pipeline_auto_schedule(input, output);
        output.device_sync();
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);
#endif

    convert_and_save_image(output, argv[2]);

    printf("Success!\n");
    return 0;
}
