// The CPU side of test/solo_retro_gpu.cu, compiled by g++ (nvcc cannot compile the
// game's headers): the cluster solver and its canonical form, behind plain functions.
#include "solo_region.h"
#include "magic.h"

namespace {
solo::Solver& cpu_solver() { static solo::Solver s; return s; }
}
void cpu_init() { magic::init(); }
// The CPU cluster solver's value (the reference of the checks).
int cpu_value(unsigned long long free, unsigned long long pieces) { return cpu_solver().value(free, pieces); }
// The CPU's canonical form.
void cpu_canonical(unsigned long long free, unsigned long long pieces, unsigned long long* f, unsigned long long* p) {
    const solo::Key k = solo::canonical(free, pieces);
    *f = k.free;
    *p = k.pieces;
}
