// This file defines our cost model as a Halide generator. It is
// templated such that it can be compiled in either forward or
// backwards mode, for inference or training respectively.

#include <utility>

#include "Halide.h"

#include "NetworkSize.h"
#include "cost_model_schedule.h"

using namespace Halide;
using Halide::Derivative;

// A model weight is either just an input, or an input and an output
// (the updated weights and the ADAM state) depending on whether we're
// doing inference or training.
template<bool training>
struct ModelWeight;

template<>
struct ModelWeight<false> : public GeneratorInput<Buffer<float>> {
    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
    }
    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
        }
    }
};

template<>
struct ModelWeight<true> : public GeneratorInput<Buffer<float>> {
    GeneratorOutput<Buffer<float>> grad;

    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim), grad("updated_" + name, dim + 1) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
        std::vector<Expr> args(dimensions() + 1);
        for (auto &e : args) {
            e = Var();
        }
        grad(args) = undef<float>();

        // We'll report back the new weights and the loss gradients,
        // and update the ADAM state. Depending on the mode the caller
        // is in, it may use the new weights, or it may just send the
        // loss gradients up to an ADAM server.
        args.back() = 0;
        FuncRef new_weight = grad(args);
        args.back() = 1;
        FuncRef smoothed_deriv = grad(args);
        args.back() = 2;
        FuncRef smoothed_second_moment = grad(args);
        args.back() = 3;
        FuncRef loss_gradient = grad(args);

        args.pop_back();
        Expr current_weight = (*this)(args);

        loss_gradient = d(*this)(args);

        // Update the first and second moment estimates
        smoothed_deriv = 0.9f * smoothed_deriv + 0.1f * loss_gradient;
        smoothed_second_moment = 0.999f * smoothed_second_moment + 0.001f * pow(loss_gradient, 2);

        // Correction to account for the fact that the smoothed_deriv
        // and smoothed_second_moment start at zero when t == 0
        Expr smoothed_deriv_correction = 1 / (1 - pow(0.9f, timestep + 1));
        Expr smoothed_second_moment_correction = 1 / (1 - pow(0.999f, timestep + 1));

        // Update the weights
        Expr step = learning_rate * smoothed_deriv * smoothed_deriv_correction;
        step /= sqrt(smoothed_second_moment * smoothed_second_moment_correction) + 1e-5f;

        new_weight = current_weight - step;
    }

    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
            dim(0).set_estimate(0, s0);
            grad.dim(0).set_bounds(0, s0);
            grad.dim(0).set_estimate(0, s0);
            grad.bound(grad.args()[0], 0, s0);
            grad.set_estimate(grad.args()[0], 0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
            dim(1).set_estimate(0, s1);
            grad.dim(1).set_bounds(0, s1);
            grad.dim(1).set_estimate(0, s1);
            grad.bound(grad.args()[1], 0, s1);
            grad.set_estimate(grad.args()[1], 0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
            dim(2).set_estimate(0, s2);
            grad.dim(2).set_bounds(0, s2);
            grad.dim(2).set_estimate(0, s2);
            grad.bound(grad.args()[2], 0, s2);
            grad.set_estimate(grad.args()[2], 0, s2);
        }
        grad.dim(dimensions()).set_bounds(0, 4);
        grad.dim(dimensions()).set_estimate(0, 4);
    }
};

template<bool training>
class CostModel : public Generator<CostModel<training>> {
public:
    template<typename T>
    using Input = GeneratorInput<T>;
    template<typename T>
    using Output = GeneratorOutput<T>;
    using Generator<CostModel<training>>::auto_schedule;
    using Generator<CostModel<training>>::get_pipeline;

    // Number of pipeline stages
    Input<int> num_stages{"num_stages", 1};

    // Batch size. Every item in the batch is a different schedule for
    // the same algorithm.
    Input<int> batch_size{"batch_size", 1};

    // Number of cores on the target machine. Used to reason about idle cores.
    Input<int> num_cores{"num_cores", 1};

    // Algorithm-specific features
    Input<Buffer<float>> pipeline_features{"pipeline_features", 3};

    // Schedule-specific features
    Input<Buffer<float>> schedule_features{"schedule_features", 3};

    // Network weights. We use some template-fu so that they are
    // inputs in inference mode, and inputs and outputs in training
    // mode.
    using Weight = ModelWeight<training>;
    Weight head1_filter{"head1_filter", 3};
    Weight head1_bias{"head1_bias", 1};
    Weight head2_filter{"head2_filter", 2};
    Weight head2_bias{"head2_bias", 1};
    Weight filter1{"filter1", 2};
    Weight bias1{"bias1", 1};

    // Some extra inputs for training mode.
    Input<float> learning_rate{"learning_rate", 1.0f};
    Input<int> timestep{"timestep", 0};  // Needed by ADAM

    // The index of the fastest schedule in the batch. Used as a
    // reference point for computing relative throughput.
    Input<int> reference{"reference", 0};

    Input<float> alpha{"alpha", 0.001f};

    // The true runtimes obtained by benchmarking.
    Input<Buffer<float>> true_runtime{"true_runtime", 1};

    // The predicted runtimes
    Output<Buffer<float>> prediction_output{"prediction_output", 1};

    Output<Buffer<float>> lower_bound_prediction_output{"lower_bound_prediction_output", 1};

    Output<Buffer<float>> upper_bound_prediction_output{"upper_bound_prediction_output", 1};

    // The loss. L2 on relative throughput.
    Output<Buffer<float>> loss_output{"loss_output", 0};

    Output<Buffer<float>> lower_bound_loss_term1_output{"lower_bound_loss_term1_output", 0};

    Output<Buffer<float>> lower_bound_loss_term2_output{"lower_bound_loss_term2_output", 0};

    Output<Buffer<float>> upper_bound_loss_term1_output{"upper_bound_loss_term1_output", 0};

    Output<Buffer<float>> upper_bound_loss_term2_output{"upper_bound_loss_term2_output", 0};



    // Zero pad alone the last dimension of a Func
    Func pad_stages(const Func &f, const Expr &stages) {
        Halide::Region bounds(f.dimensions());
        bounds[1].min = 0;
        bounds[1].extent = stages;
        return BoundaryConditions::constant_exterior(f, cast(f.value().type(), 0), bounds);
    }

    Expr activation(const Expr &e) {
        // relu
        return max(e, 0);
    }

    Expr sigmoid(const Expr &e) {
        return 1 / (1 + exp(-e));
    }

    void generate() {
        Var c("c"), w("w"), n("n"), j("j"), s("s");

        Func normalized_schedule_features("normalized_schedule_features");
        normalized_schedule_features(n, c, s) = fast_log(schedule_features(n, c, s) + 1);

        // Force the weights of the algorithm embedding layer to be positive and bounded.
        Func squashed_head1_filter("squashed_head1_filter");
        squashed_head1_filter(c, s, n) = sigmoid(head1_filter(c, s, n));

        // Explicitly broadcast the weights across the batch. This
        // give the autoscheduler some more options in the
        // reverse-mode pipeline.
        Func squashed_head1_filter_broadcast("squashed_head1_filter_broadcast");
        squashed_head1_filter_broadcast(c, w, s, n) = squashed_head1_filter(c, s, n);

        // The conv layer that embeds the algorithm-specific features.
        Func head1_conv("head1_conv");
        RDom r_head1(0, head1_w, 0, head1_h);
        head1_conv(c, w) = head1_bias(c);
        head1_conv(c, w) += (squashed_head1_filter_broadcast(c, w, r_head1.x, r_head1.y) *
                             pipeline_features(r_head1.x, r_head1.y, w));

        // No point in a relu - the inputs and weights are positive

        // The conv layer that embeds the schedule-specific features.
        Func head2_conv("head2_conv");
        RDom r_head2(0, head2_w);
        head2_conv(c, w, n) = head2_bias(c);
        head2_conv(c, w, n) += head2_filter(c, r_head2) * normalized_schedule_features(n, r_head2, w);

        Func head2_relu("head2_relu");
        head2_relu(c, w, n) = activation(head2_conv(c, w, n));

        // The conv layer that computes coefficients, split into two
        // stages. First we consumer the algorithm embedding.
        Func conv1_stage1("conv1_stage1");
        RDom r1_stage1(0, head1_channels);
        conv1_stage1(c, w) = bias1(c);
        conv1_stage1(c, w) += filter1(c, r1_stage1.x) * head1_conv(r1_stage1.x, w);

        // Then we consume the schedule embedding.
        Func conv1_stage2("conv1_stage2");
        RDom r1_stage2(0, head2_channels);
        conv1_stage2(c, w, n) = conv1_stage1(c, w);
        conv1_stage2(c, w, n) += filter1(c, head1_filter.dim(0).extent() + r1_stage2.x) * head2_relu(r1_stage2.x, w, n);

        // The final set of predicted coefficients.
        Func relu1("relu1");
        relu1(c, w, n) = activation(conv1_stage2(c, w, n));

        // That's the end of the neural network. Now we will use these
        // coefficients with a bunch of hand-designed terms.

        // Unpack all of the schedule features. We don't use all of
        // them, but it's easier to avoid bugs if we just unpack them
        // all in the same order as Featurization.h
        int idx = 0;
        Expr num_realizations = schedule_features(n, idx++, w);
        Expr num_productions = schedule_features(n, idx++, w);
        Expr points_computed_per_realization = schedule_features(n, idx++, w);
        Expr points_computed_per_production = schedule_features(n, idx++, w);
        Expr points_computed_total = schedule_features(n, idx++, w);
        Expr points_computed_minimum = schedule_features(n, idx++, w);
        Expr innermost_loop_extent = schedule_features(n, idx++, w);
        Expr innermost_pure_loop_extent = schedule_features(n, idx++, w);
        Expr unrolled_loop_extent = schedule_features(n, idx++, w);
        Expr inner_parallelism = schedule_features(n, idx++, w);
        Expr outer_parallelism = schedule_features(n, idx++, w);
        Expr bytes_at_realization = schedule_features(n, idx++, w);
        Expr bytes_at_production = schedule_features(n, idx++, w);
        Expr bytes_at_root = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_realization = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_production = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_root = schedule_features(n, idx++, w);
        Expr inlined_calls = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_realization = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_realization = schedule_features(n, idx++, w);
        Expr allocation_bytes_read_per_realization = schedule_features(n, idx++, w);
        Expr working_set = schedule_features(n, idx++, w);
        Expr vector_size = schedule_features(n, idx++, w);
        Expr native_vector_size = schedule_features(n, idx++, w);
        Expr num_vectors = schedule_features(n, idx++, w);
        Expr num_scalars = schedule_features(n, idx++, w);
        Expr scalar_loads_per_vector = schedule_features(n, idx++, w);
        Expr vector_loads_per_vector = schedule_features(n, idx++, w);
        Expr scalar_loads_per_scalar = schedule_features(n, idx++, w);
        Expr bytes_at_task = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_task = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_vector = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_vector = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_task = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_task = schedule_features(n, idx++, w);
        Expr working_set_at_task = schedule_features(n, idx++, w);
        Expr working_set_at_production = schedule_features(n, idx++, w);
        Expr working_set_at_realization = schedule_features(n, idx++, w);
        Expr working_set_at_root = schedule_features(n, idx++, w);
        assert(idx == head2_w);

        // Count up the number of things computed, applying a
        // different cost to vectors and scalars, and a different cost
        // depending on whether we were inlined.
        Expr lower_compute_cost = select(inlined_calls == 0,
                                   (vector_size * num_vectors * relu1(0, w, n) +
                                    num_scalars * relu1(1, w, n)),
                                   (vector_size * num_vectors * relu1(2, w, n) +
                                    num_scalars * relu1(3, w, n)));

        Expr upper_compute_cost = select(inlined_calls == 0,
                                   (vector_size * num_vectors * relu1(32, w, n) +
                                    num_scalars * relu1(33, w, n)),
                                   (vector_size * num_vectors * relu1(34, w, n) +
                                    num_scalars * relu1(35, w, n)));

        // Round up these costs according to how neatly we're using
        // our cores.
        Expr num_tasks = max(1, inner_parallelism * outer_parallelism);
        Expr tasks_per_core = num_tasks / num_cores;
        Expr idle_core_wastage = ceil(tasks_per_core) / max(1, tasks_per_core);
        lower_compute_cost *= idle_core_wastage;
        upper_compute_cost *= idle_core_wastage;

        // Next comes a long list of plausible terms to capture the cost of loads.
        Expr lower_load_cost = (num_realizations * unique_lines_read_per_realization * relu1(5, w, n) +
                          num_realizations * unique_bytes_read_per_realization * relu1(6, w, n) +
                          num_vectors * scalar_loads_per_vector * relu1(7, w, n) +
                          num_scalars * scalar_loads_per_scalar * relu1(8, w, n) +
                          num_vectors * vector_loads_per_vector * relu1(9, w, n) +
                          num_scalars * unique_bytes_read_per_vector * relu1(10, w, n) +
                          num_vectors * unique_bytes_read_per_vector * relu1(11, w, n) +
                          num_scalars * unique_lines_read_per_vector * relu1(12, w, n) +
                          num_vectors * unique_lines_read_per_vector * relu1(13, w, n) +
                          num_tasks * unique_bytes_read_per_task * relu1(14, w, n) +
                          num_tasks * unique_lines_read_per_task * relu1(15, w, n));

        Expr upper_load_cost = (num_realizations * unique_lines_read_per_realization * relu1(37, w, n) +
                          num_realizations * unique_bytes_read_per_realization * relu1(38, w, n) +
                          num_vectors * scalar_loads_per_vector * relu1(39, w, n) +
                          num_scalars * scalar_loads_per_scalar * relu1(40, w, n) +
                          num_vectors * vector_loads_per_vector * relu1(41, w, n) +
                          num_scalars * unique_bytes_read_per_vector * relu1(42, w, n) +
                          num_vectors * unique_bytes_read_per_vector * relu1(43, w, n) +
                          num_scalars * unique_lines_read_per_vector * relu1(44, w, n) +
                          num_vectors * unique_lines_read_per_vector * relu1(45, w, n) +
                          num_tasks * unique_bytes_read_per_task * relu1(46, w, n) +
                          num_tasks * unique_lines_read_per_task * relu1(47, w, n));

        // Next we have the cost of stores.
        Expr lines_written_per_realization = inner_parallelism * (bytes_at_task / max(1, innermost_bytes_at_task));

        // Use separate coefficients for things with internal
        // parallelism, because for stages with internal parallelism,
        // most values of the values being stored will be consumed on
        // another core, so they will get punted out to L3 no matter
        // how small. Also use a separate term for the final stage, as
        // we never pay the cost of loading from it.
        Expr lower_alpha = select(inner_parallelism > 1, relu1(16, w, n),
                            w == 0, relu1(17, w, n),
                            relu1(18, w, n));
        Expr lower_beta = select(inner_parallelism > 1, relu1(19, w, n),
                           w == 0, relu1(20, w, n),
                           relu1(21, w, n));

        Expr upper_alpha = select(inner_parallelism > 1, relu1(48, w, n),
                            w == 0, relu1(49, w, n),
                            relu1(50, w, n));
        Expr upper_beta = select(inner_parallelism > 1, relu1(51, w, n),
                           w == 0, relu1(52, w, n),
                           relu1(53, w, n));


        Expr lower_store_cost = num_realizations * (lines_written_per_realization * lower_alpha +
                                                    bytes_at_realization * lower_beta);
        Expr upper_store_cost = num_realizations * (lines_written_per_realization * upper_alpha +
                                                    bytes_at_realization * upper_beta);

        // Now account for false sharing of cache lines. The
        // probability of a store hitting a cache line also hit by
        // another core is inversely proportional to
        // innermost_bytes_at_task, and the cost is paid on every
        // store.
        Expr lower_cost_of_false_sharing =
            select(inner_parallelism > 1,
                   relu1(22, w, n) * (num_vectors + num_scalars) / max(1, innermost_bytes_at_task),
                   0.0f);

        Expr upper_cost_of_false_sharing =
            select(inner_parallelism > 1,
                   relu1(54, w, n) * (num_vectors + num_scalars) / max(1, innermost_bytes_at_task),
                   0.0f);

        lower_store_cost += lower_cost_of_false_sharing;
        upper_store_cost += upper_cost_of_false_sharing;

        // Now add a term for false sharing of pages. The maximum
        // number of threads that could all fault on the same page at
        // the same time is:
        Expr max_threads_hitting_same_page_fault = min(inner_parallelism, 4096 / max(1, innermost_bytes_at_task));

        // The total number of page faults is proportionate to the number of bytes allocated
        const Expr &num_page_faults = bytes_at_production;

        // And page faults are serviced serially, so the total CPU time gets multiplied by the thread count again!
        Expr lower_cost_of_page_faults = (num_page_faults * max_threads_hitting_same_page_fault *
                                    inner_parallelism * outer_parallelism * relu1(23, w, n));
        Expr upper_cost_of_page_faults = (num_page_faults * max_threads_hitting_same_page_fault *
                                    inner_parallelism * outer_parallelism * relu1(55, w, n));

        lower_store_cost += lower_cost_of_page_faults;
        upper_store_cost += upper_cost_of_page_faults;

        // Malloc is not free, so add a cost per allocation.
        Expr lower_cost_of_malloc = relu1(24, w, n) * num_realizations;
        Expr upper_cost_of_malloc = relu1(56, w, n) * num_realizations;

        // A cost for launching a parallel task...
        Expr lower_cost_of_parallel_launches = num_productions * select(inner_parallelism > 1, relu1(25, w, n), 0.0f);
        Expr upper_cost_of_parallel_launches = num_productions * select(inner_parallelism > 1, relu1(57, w, n), 0.0f);

        // ... and an overhead per task.
        Expr lower_cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * relu1(26, w, n);
        Expr upper_cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * relu1(58, w, n);

        Expr lower_cost_of_parallelism = lower_cost_of_parallel_launches + lower_cost_of_parallel_tasks;
        Expr upper_cost_of_parallelism = upper_cost_of_parallel_launches + upper_cost_of_parallel_tasks;

        // Make it easier for the model to penalize working sets that
        // start to fall out of cache by giving it a term that gets
        // multiplied by the working set.
        Expr lower_cost_of_working_set = working_set * relu1(27, w, n);
        Expr upper_cost_of_working_set = working_set * relu1(59, w, n);

        // FIXME: For our best set of trained weights, store_cost was
        // accidentally in the list below twice, so we double it here
        // in order to not have to retrain.
        lower_store_cost *= 2;
        upper_store_cost *= 2;


        Expr lower_bound_cost = (lower_compute_cost +
                                lower_store_cost +
                                lower_load_cost +
                                lower_cost_of_malloc +
                                lower_cost_of_parallelism +
                                lower_cost_of_working_set);

        Expr upper_bound_cost = (upper_compute_cost +
                                upper_store_cost +
                                upper_load_cost +
                                upper_cost_of_malloc +
                                upper_cost_of_parallelism +
                                upper_cost_of_working_set);

        for (int i = 0; i < 64; i++) {
            lower_bound_cost += 0.0f * relu1(i, w, n);
        }

        Func runtime_per_stage;
        // Change units so that network weights are in a human-readable range.
        runtime_per_stage(n, w) = lower_bound_cost * 1e-9f;

        Func lower_bound_runtime_per_stage;
        lower_bound_runtime_per_stage(n, w) = lower_bound_cost * 1e-9f;

        Func upper_bound_runtime_per_stage;
        upper_bound_runtime_per_stage(n, w) = upper_bound_cost * 1e-9f;

        // Sum across the stages.
        Func prediction;
        Func lower_bound_prediction;
        Func upper_bound_prediction;
        RDom r_reduce(0, num_stages);
        RDom lr_reduce(0, num_stages);
        RDom ur_reduce(0, num_stages);
        prediction(n) += runtime_per_stage(n, r_reduce);
        lower_bound_prediction(n) += lower_bound_runtime_per_stage(n, lr_reduce);
        upper_bound_prediction(n) += upper_bound_runtime_per_stage(n, ur_reduce);
        

        prediction_output(n) = prediction(n);
        lower_bound_prediction_output(n) = lower_bound_prediction(n);
        upper_bound_prediction_output(n) = upper_bound_prediction(n);

        Func err;
        Func lower_err1, lower_err2, upper_err1, upper_err2;

        if (!training) {
            loss_output() = 0.0f;
            lower_bound_loss_term1_output() = 0.0f;
            lower_bound_loss_term2_output() = 0.0f;
            upper_bound_loss_term1_output() = 0.0f;
            upper_bound_loss_term2_output() = 0.0f;
        } else {

            // The tail end of the reverse-mode pipeline
            RDom r_batch(0, batch_size);

            // We believe the coefficients on all the various
            // components of cost should be positive, even before the
            // relu, and even before schedule-specific features are
            // taken into account. The network shouldn't be telling us
            // that things would be cheaper if we would do more
            // mallocs, or compute more values, or launch more
            // parallel tasks. So we add a regularization term. This
            // helps dead relus get unstuck.
            RDom r_conv1_output(0, conv1_channels, 0, num_stages);
            Expr regularize = sum(-min(conv1_stage2(r_conv1_output.x, r_conv1_output.y, n), 0));

            // Our loss will be L2 on relative throughput.

            // Get the reference runtime.
            Expr n2 = clamp(reference, 0, batch_size - 1);
            Expr scale = 1.0f / true_runtime(n2);

            // Compute the relative true runtime and the relative predicted runtime
            Expr r1 = true_runtime(n) * scale;
            Expr lower_bound_p1 = lower_bound_prediction(n) * scale;
            Expr upper_bound_p1 = upper_bound_prediction(n) * scale;

            // Invert them to get relative throughput, and compute L2 loss.
            Expr lower_bound_delta_term1 = pow(max(0.0f, 1.0f / r1 - 1.0f / max(lower_bound_p1, 1e-10f)), 2); 
            Expr lower_bound_delta_term2 = alpha * pow((1.0f / max(lower_bound_p1, 1e-10f) - 1.0f / r1), 2);
            Expr upper_bound_delta_term1 = pow(max(0.0f, 1.0f / max(upper_bound_p1, 1e-10f) - 1.0f / r1), 2);
            Expr upper_bound_delta_term2 = alpha * pow((1.0f / max(upper_bound_p1, 1e-10f) - 1.0f / r1), 2);
            Expr delta = lower_bound_delta_term1 + lower_bound_delta_term2 + upper_bound_delta_term1 + upper_bound_delta_term2;
            // Add the regulization with a small weight.
            err(n) = delta + 1e-5f * regularize;
            lower_err1(n) = lower_bound_delta_term1 + 1e-5f * regularize;
            lower_err2(n) = lower_bound_delta_term2 + 1e-5f * regularize;
            upper_err1(n) = upper_bound_delta_term1 + 1e-5f * regularize;
            upper_err2(n) = upper_bound_delta_term2 + 1e-5f * regularize;


            // Sum the errors over the batch.
            Expr loss = sum(err(r_batch));
            Expr lower_term1_loss = sum(lower_err1(r_batch));
            Expr lower_term2_loss = sum(lower_err2(r_batch));
            Expr upper_term1_loss = sum(upper_err1(r_batch));
            Expr upper_term2_loss = sum(upper_err2(r_batch));

            loss_output() = loss;
            lower_bound_loss_term1_output() = lower_term1_loss;
            lower_bound_loss_term2_output() = lower_term2_loss;
            upper_bound_loss_term1_output() = upper_term1_loss;
            upper_bound_loss_term2_output() = upper_term2_loss;

            // Compute derivatives of the loss, and backpropagate them
            // to the model weights.
            Derivative d_loss_d = propagate_adjoints(loss_output);

            Weight *weights[] = {&head1_filter, &head1_bias,
                                 &head2_filter, &head2_bias,
                                 &filter1, &bias1};

            for (Weight *w : weights) {
                w->backprop(d_loss_d, learning_rate, timestep);
            }
        }

        // All the model weight shapes are statically known, so we
        // tell Halide their sizes to simplify the generated code.
        head1_filter.set_shape(head1_channels, head1_w, head1_h);
        head1_bias.set_shape(head1_channels);
        head2_filter.set_shape(head2_channels, head2_w);
        head2_bias.set_shape(head2_channels);
        filter1.set_shape(conv1_channels, head1_channels + head2_channels);
        bias1.set_shape(conv1_channels);

        // Estimates for autoscheduling this pipeline (using
        // itself!). We do that offline and check in the generated
        // schedule source, so that bugs in our autoscheduler don't
        // cause build nightmares due to the circular dependency.
        num_cores.set_estimate(32);
        reference.set_estimate(0);
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        prediction_output.set_estimates({{0, 80}});
        lower_bound_prediction_output.set_estimates({{0, 80}});
        upper_bound_prediction_output.set_estimates({{0, 80}});
        learning_rate.set_estimate(0.001f);
        timestep.set_estimate(37);
        pipeline_features.set_estimates({{0, head1_w}, {0, head1_h}, {0, 13}});
        schedule_features.set_estimates({{0, 80}, {0, head2_w}, {0, 13}});
        true_runtime.set_estimates({{0, 80}});

        // SCHEDULE
        if (training && !auto_schedule) {
            do_cost_model_schedule(get_pipeline());
        } else if (auto_schedule) {
            // Do nothing.
        } else {
            // We just write down a good schedule for
            // inference. Scheduling a couple of convs is easy.
            Var no;
            prediction_output.specialize(batch_size < 8).split(n, no, n, 1);
            prediction_output.compute_root().split(n, no, n, 8).parallel(no);
            prediction_output.bound(n, 0, batch_size);

            // schedule for the forwards path
            const int vec = 8;

            // A helper function for scheduling conv layers
            auto schedule_conv = [&](Func conv, Func relu, const RVar &r_channels) {
                Var ci, wi;
                if (!training) {
                    relu
                        .compute_at(prediction_output, n)
                        .store_at(prediction_output, no)
                        .tile(c, w, ci, wi, vec, 4, TailStrategy::RoundUp)
                        .vectorize(ci);
                    conv.compute_at(relu, c);
                } else {
                    // In training mode, we need the conv activations pre-relu too
                    conv.in()
                        .compute_root()
                        .tile(c, w, ci, wi, vec, 1, TailStrategy::RoundUp)
                        .vectorize(ci)
                        .unroll(wi)
                        .parallel(n, 8);
                    conv.compute_at(conv.in(), c);
                    relu
                        .compute_root()
                        .reorder_storage(c, w, n)
                        .reorder(c, w, n)
                        .vectorize(c, vec)
                        .parallel(n, 8);
                }
                conv
                    .vectorize(c)
                    .unroll(w)
                    .update()
                    .vectorize(c)
                    .unroll(w)
                    .reorder(c, w, r_channels);
            };

            // Pipeline features processing
            conv1_stage1.compute_root().vectorize(c).update().vectorize(c);
            squashed_head1_filter.compute_root().vectorize(c);

            // Schedule features processing. The number of schedule
            // features is not close to a multiple of 8, so vectorized
            // across the batch.
            if (!training) {
                normalized_schedule_features
                    .compute_at(prediction_output, no)
                    .vectorize(n);
            } else {
                normalized_schedule_features
                    .compute_root()
                    .vectorize(n, 8);
            }

            // conv+relu layers
            schedule_conv(head2_conv, head2_relu, r_head2.x);
            schedule_conv(conv1_stage2, relu1, r1_stage2.x);
        }
    }
};

using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model);
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model);
