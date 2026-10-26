#ifndef AMREX_MF_PARALLEL_FOR_H_
#define AMREX_MF_PARALLEL_FOR_H_
#include <AMReX_Config.H>

#include <AMReX_Concepts.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_TypeTraits.H>

#ifdef AMREX_USE_GPU
#include <AMReX_MFParallelForG.H>
#else
#include <AMReX_MFParallelForC.H>
#endif

namespace amrex {

struct TileSize {
    IntVect tile_size;
    explicit TileSize (IntVect const& ts) noexcept : tile_size(ts) {}
};

struct DynamicTiling {
    bool dynamic;
    explicit DynamicTiling (bool f) noexcept : dynamic(f) {}
};

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid region.  If built for
 * CPU, tiling will be enabled.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, F&& f)
{
    detail::ParallelFor_doit(mf, IntVect(0), FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid region.  If built for
 * CPU, tiling will be enabled.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, IntVect(0), 1, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, IntVect(0), FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, F&& f)
{
    detail::ParallelFor_doit(mf, ng, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, 1, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, F&& f)
{
    detail::ParallelFor_doit(mf, ng, ncomp, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, ncomp, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, ncomp, FabArrayBase::mfiter_tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid region.  If built for
 * CPU, tiling will be enabled.  However, one could specify a huge tile size
 * to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, TileSize const& ts, F&& f)
{
    detail::ParallelFor_doit(mf, IntVect(0), ts.tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid region.  If built for
 * CPU, tiling will be enabled.  However, one could specify a huge tile size
 * to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, TileSize const& ts, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, IntVect(0), 1, ts.tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, IntVect(0), ts.tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, TileSize const& ts, F&& f)
{
    detail::ParallelFor_doit(mf, ng, ts.tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, TileSize const& ts, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, 1, ts.tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, ts.tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, TileSize const& ts, F&& f)
{
    detail::ParallelFor_doit(mf, ng, ncomp, ts.tile_size, false, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param ts tile size, ignored by GPU build
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, TileSize const& ts, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, ncomp, ts.tile_size, false, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, ncomp, ts.tile_size, false, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ts tile size, ignored by GPU build
 * \param dt controls dynamic tiling for the cpu build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, TileSize const& ts, DynamicTiling dt, F&& f)
{
    detail::ParallelFor_doit(mf, ng, ts.tile_size, dt.dynamic, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param ts tile size, ignored by GPU build
 * \param dt controls dynamic tiling for the cpu build
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, TileSize const& ts,
             DynamicTiling dt, F&& f)
{
    detail::ParallelFor_doit(mf, ng, ncomp, ts.tile_size, dt.dynamic, std::forward<F>(f));
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For GPU builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 4D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ts tile size, ignored by GPU build
 * \param dt controls dynamic tiling for the cpu build
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, TileSize const& ts,
             DynamicTiling dt, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, 1, ts.tile_size, dt.dynamic, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, ts.tile_size, dt.dynamic, std::forward<F>(f));
#endif
}

/**
 * \brief ParallelFor for MultiFab/FabArray.
 *
 * This version launches a kernel to work on the valid and ghost regions.  If
 * built for CPU, tiling will be enabled.  However, one could specify a huge
 * tile size to effectively disable tiling.  For gpu builds, this function is
 * NON-BLOCKING on the host. Conceptually, this is a 5D loop.
 *
 * \tparam MT max threads in GPU blocks (Only relevant for GPU builds)
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ng the number of ghost cells around the valid region
 * \param ncomp the number of component
 * \param ts tile size, ignored by GPU build
 * \param dt controls dynamic tiling for the cpu build
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <int MT, FabArrayType MF, typename F>
void
ParallelFor (MF const& mf, IntVect const& ng, int ncomp, TileSize const& ts,
             DynamicTiling dt, F&& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit<MT>(mf, ng, ncomp, ts.tile_size, dt.dynamic, std::forward<F>(f));
#else
    detail::ParallelFor_doit(mf, ng, ncomp, ts.tile_size, dt.dynamic, std::forward<F>(f));
#endif
}

/**
 * \brief Index offset of one color of a multi-color sweep.
 *
 * Bit d of \p color is the index parity in direction d, so the 2^DIM colors
 * partition the points. Use with ParallelForStrided and a stride of 2.
 */
[[nodiscard]] constexpr IntVect multicolor_offset (int color) noexcept
{
    return IntVect(AMREX_D_DECL(color&1, (color>>1)&1, (color>>2)&1));
}

/**
 * \brief One-pass multi-color sweep of a box on the CPU.
 *
 * Visits the 2^DIM colors of multicolor_offset in color order and gives
 * every point the same neighbor values as 2^DIM separate color passes do
 * (stride 2, stencil reach 1): color c sweeps the plane (row in 2D, point
 * in 1D) that lags color 0 by c + offset[last] planes in the last
 * direction, the smallest lag at which colors < c are finished and colors
 * > c untouched on the neighboring planes (colors of the same parity in
 * that direction meet only on one plane, so they need a lag of one).  At
 * most 2^DIM + 1 planes are live, and the box is streamed once per sweep
 * instead of 2^DIM times.  The caller must own the whole box: neighboring
 * regions swept concurrently (e.g., OpenMP tiles) would be at different
 * colors.
 */
template <class F>
void LoopMulticolorOnCpu (Box const& bx, F const& f) noexcept
{
    constexpr int ncolors = 1 << AMREX_SPACEDIM;
    constexpr int last = AMREX_SPACEDIM-1;
    const IntVect lo = bx.smallEnd(), hi = bx.bigEnd();
    for (int step = lo[last]; step <= hi[last] + ncolors; ++step) { // largest lag is ncolors
        for (int color = 0; color < ncolors; ++color) {
            IntVect const offset = multicolor_offset(color);
            IntVect clo = lo, chi = hi;
            clo[last] = chi[last] = step - (color + offset[last]);
            if (clo[last] < lo[last] || clo[last] > hi[last] ||
                ((clo[last]-offset[last]) & 1)) { continue; } // not a plane of this color
            for (int d = 0; d < last; ++d) { clo[d] += (offset[d]-lo[d]) & 1; } // first node of this color
            const Dim3 l = clo.dim3(0), h = chi.dim3(0);
            for (        int k = l.z; k <= h.z; k += 2) {
                for (    int j = l.y; j <= h.y; j += 2) {
                    for (int i = l.x; i <= h.x; i += 2) {
                        f(i,j,k);
                    }
                }
            }
        }
    }
}

/**
 * \brief ParallelFor over a strided subset of the valid region of a MultiFab/FabArray.
 *
 * Only points whose index in direction d equals offset[d] modulo stride[d]
 * are visited, e.g., one color of a multi-color Gauss-Seidel sweep.  For
 * GPU builds, all boxes are covered by a single kernel launch, and this
 * function is NON-BLOCKING on the host.  For CPU builds, tiles run in
 * parallel with OpenMP, so a Gauss-Seidel sweep is safe only if the points
 * visited together are not coupled by the stencil.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param stride positive stride in each direction
 * \param offset index offset in each direction
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelForStrided (MF const& mf, IntVect const& stride, IntVect const& offset, F&& f)
{
    AMREX_ASSERT(stride.allGT(0));
    // Equivalent offsets share one cached ParForInfo.
    IntVect o;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        o[d] = ((offset[d] % stride[d]) + stride[d]) % stride[d];
    }
    detail::ParallelForStrided_doit(mf, stride, o, std::forward<F>(f));
}

/**
 * \brief ParallelFor over the red or black points of the valid region of a MultiFab/FabArray.
 *
 * Only points (i,j,k) with i+j+k+redblack even are visited, e.g., one half
 * of a red-black Gauss-Seidel sweep.  For GPU builds, all boxes are covered
 * by a single kernel launch, and this function is NON-BLOCKING on the host.
 * For CPU builds, tiles run in parallel with OpenMP, so a Gauss-Seidel
 * sweep is safe only if points of the same color are not coupled by the
 * stencil.
 *
 * \tparam MF the MultiFab/FabArray type
 * \tparam F a callable type like lambda
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param redblack 0 or 1, selecting the red or black points
 * \param f a callable object void(int,int,int,int), where the first argument
 *           is the local box index, and the following three are spatial indices
 *           for x, y, and z-directions.
 */
template <FabArrayType MF, typename F>
void
ParallelForRedBlack (MF const& mf, int redblack, F&& f)
{
    detail::ParallelForRedBlack_doit(mf, redblack, std::forward<F>(f));
}

/**
 * \brief ParallelFor over the red or black points of the valid region of a MultiFab/FabArray.
 *
 * Same as above, but f is called for each of the \p ncomp components.
 *
 * \param mf the MultiFab/FabArray object used to specify the iteration space
 * \param ncomp the number of components
 * \param redblack 0 or 1, selecting the red or black points
 * \param f a callable object void(int,int,int,int,int), where the first argument
 *           is the local box index, the following three are spatial indices
 *           for x, y, and z-directions, and the last is for component.
 */
template <FabArrayType MF, typename F>
void
ParallelForRedBlack (MF const& mf, int ncomp, int redblack, F const& f)
{
    if (ncomp == 1) {
        detail::ParallelForRedBlack_doit(mf, redblack,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            f(box_no, i, j, k, 0);
        });
    } else {
        detail::ParallelForRedBlack_doit(mf, redblack,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            for (int n = 0; n < ncomp; ++n) {
                f(box_no, i, j, k, n);
            }
        });
    }
}

}

#endif
