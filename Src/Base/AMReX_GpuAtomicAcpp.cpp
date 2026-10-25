#include <AMReX_Config.H>

// Definition of the flag that AMReX_GpuAtomic.H reads in AdaptiveCpp kernels.
// This file must not include AMReX_GpuAtomic.H: there acpp_cpu_serial() is
// declared weak and const for its callers in kernels, and this definition
// reads a variable, so it must be neither.

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)

namespace amrex::Gpu::Atomic::detail {

namespace {
    int cpu_serial = 0;
}

int acpp_cpu_serial ()
{
    return cpu_serial;
}

void acpp_set_cpu_serial (bool on) noexcept
{
    cpu_serial = on ? 1 : 0;
}

}

#endif
