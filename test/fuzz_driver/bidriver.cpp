#include "Halide.h"
#include <fstream>
#include <set>

using namespace Halide;

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

    // Schedule
// input_im.rename(_1, _1rn).fuse(_0, _1rn, _0f_1rn).reorder(_0f_1rn)

// lambda_0.rename(_1, _1rnrn).fuse(_0, _1rnrn, _0f_1rnrnrnrn).reorder(_0f_1rnrnrnrn)

// repeat_edge.fuse(_1, _0, _1f_0).split(_1f_0, _1f_0o, _1f_0i, 3, TailStrategy::RoundUp).reorder(_1f_0o, _1f_0i)

    // Var xo, vxi, xrn, xrnfc, xrnfcorn, xrnfci, crn, crno, crni, crnfv2rn, cofzrn, v5, v6, xify, xoo, xoi, xoifci, xoofxoifci, yfv7o, yfv7ofv8;
    // Var xrnrn, v2rn, v7, v8, v7o, v7i, v7io, v7ii, co, ci, zrn, zo, vzi, yfv7ofv8rn;
    // histogram.compute_root().rename(x, xrn).fuse(xrn, c, xrnfc).split(xrnfc, xrnfcorn, xrnfci, 3, TailStrategy::ShiftInwards).reorder(xrnfci, y, xrnfcorn, z);
    // histogram.update(0).rename(c, crn).split(crn, crno, crni, 3, TailStrategy::GuardWithIf).fuse(x, r.x, xfr10$y).reorder(r10$x, xfr10$y, y, crni, crno).parallel(x.xfr10$y)

    // blurz.compute_root().split(x, x, v2rn, 5, TailStrategy::GuardWithIf).rename(c, crn).rename(z, zrn).fuse(crn, v2rn, crnfv2rn).reorder(zrn, y, crnfv2rn, x);

    // blurx.compute_root().split(x, xrnrn, v5, 8, TailStrategy::ShiftInwards).split(c, co, ci, 2, TailStrategy::ShiftInwards).fuse(co, z, cofzrn).reorder(cofzrn, xrnrn, v5, y, ci).parallel(v5).parallel(y);


    Var v6, zo, vzi;

    blury.compute_root().split(x, x, v6, 6, TailStrategy::GuardWithIf).split(z, zo, vzi, 8, TailStrategy::GuardWithIf).reorder(y, x, c, vzi, zo, v6).vectorize(vzi).vectorize(v6);

    // interpolated.compute_root().split(c, co, ci, 3, TailStrategy::PredicateLoads).split(x, xo, vxi, 5, TailStrategy::Predicate).split(xo, xoo, xoi, 16, TailStrategy::GuardWithIf).fuse(xoi, ci, xoifci).fuse(vxi, y, xify).fuse(xoo, xoifci, xoofxoifci).reorder(xify, xoofxoifci, co);

    // bilateral_grid.compute_root().split(y, y, v7, 8, TailStrategy::ShiftInwards).split(x, x, v8, 8, TailStrategy::ShiftInwards).split(v7, v7o, v7i, 5, TailStrategy::GuardWithIf).fuse(y, v7o, yfv7o).split(v7i, v7io, v7ii, 2, TailStrategy::ShiftInwards).fuse(yfv7o, v8, yfv7ofv8rn).reorder(yfv7ofv8rn, v7io, v7ii, x).parallel(yfv7ofv8rn);

    p.compile_to_module({input}, "bilateral_grid", {Target("host")});
    return 0;
}
