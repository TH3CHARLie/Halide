#ifndef HALIDE_NETWORK_SIZE_H
#define HALIDE_NETWORK_SIZE_H

namespace Halide {
// The size of the best cost model network found. Needed by the cost
// model and also the cost model training script.
const int head1_channels = 16, head1_w = 40, head1_h = 7;
const int head2_channels = 48, head2_w = 39;
const int conv1_channels = 64;
}  // namespace Halide

#endif  // HALIDE_NETWORK_SIZE_H
