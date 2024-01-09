#ifndef HALIDE_INTERNAL_SCHEDULE_MUTATOR_H
#define HALIDE_INTERNAL_SCHEDULE_MUTATOR_H

#include <random>
#include "Pipeline.h"
namespace Halide {

class ScheduleMutator {
public:
    ScheduleMutator(Pipeline p, unsigned int seed) : p(p), rng(seed) {
    }

    Pipeline mutate();

private:

    void mutate_function_schedule(Internal::Function &f);

    void mutate_loop_schedule(Internal::StageSchedule &s);

    Pipeline p;

    std::mt19937 rng;
};
}

#endif
