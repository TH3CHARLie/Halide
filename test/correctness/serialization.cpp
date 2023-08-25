#include "Halide.h"
#include <algorithm>
#include <stdio.h>

// Support code for loading pngs.
using namespace Halide;

int main(int argc, char **argv) {

    // First we'll declare some Vars to use below.
    Var x("x"), y("y"), c("c");

    // Let's create an ImageParam for an 8-bit RGB image that we'll use for input.
    ImageParam input(UInt(8), 3, "input");

    // Wrap the input in a Func that prevents reading out of bounds:
    Func clamped("clamped");
    Expr clamped_x = clamp(x, 0, input.width() - 1);
    Expr clamped_y = clamp(y, 0, input.height() - 1);
    clamped(x, y, c) = input(clamped_x, clamped_y, c);

    // Upgrade it to 16-bit, so we can do math without it overflowing.
    Func input_16("input_16");
    input_16(x, y, c) = cast<uint16_t>(clamped(x, y, c));

    // Blur it horizontally:
    Func blur_x("blur_x");
    blur_x(x, y, c) = (input_16(x - 1, y, c) +
                        2 * input_16(x, y, c) +
                        input_16(x + 1, y, c)) / 4;

    // Blur it vertically:
    Func blur_y("blur_y");
    blur_y(x, y, c) = (blur_x(x, y - 1, c) +
                        2 * blur_x(x, y, c) +
                        blur_x(x, y + 1, c)) / 4;

    // Convert back to 8-bit.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

    // Now lets serialize the pipeline to disk (must use the .hlpipe file extension)
    Pipeline blur_pipeline(output);
    std::map<std::string, Internal::Parameter> params;
    serialize_pipeline(blur_pipeline, "blur.hlpipe", params);

    // The call to serialize_pipeline populates the params map with any input or output parameters
    // that were found ... object's we'll need to attach to buffers if we wish to execute the pipeline
    for(auto named_param: params) {
        std::cout << "Found Param: " << named_param.first << std::endl;
    }

    // Lets construct a new pipeline from scratch by deserializing the file we wrote to disk
    Pipeline d_blur_pipeline = deserialize_pipeline("blur.hlpipe", params);

    return 0;

}