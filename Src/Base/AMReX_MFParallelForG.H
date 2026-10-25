#ifndef AMREX_MF_PARALLEL_FOR_G_H_
#define AMREX_MF_PARALLEL_FOR_G_H_
#include <AMReX_Config.H>

#ifdef AMREX_USE_GPU

#include <AMReX_Concepts.H>
#include <AMReX_GpuDevice.H>
#include <algorithm>
#include <cmath>
#include <limits>

/// \cond DOXYGEN_IGNORE
namespace amrex::detail {

inline
void build_par_for_boxes (char*& hp, BoxIndexer*& pboxes, Vector<Box> const& boxes)
{
    if (boxes.empty()) { return; }
    const int nboxes = boxes.size();
    const std::size_t nbytes = nboxes*sizeof(BoxIndexer);
    hp = (char*)The_Pinned_Arena()->alloc(nbytes);
    auto* hp_boxes = (BoxIndexer*)hp;
    for (int i = 0; i < nboxes; ++i) {
        new (hp_boxes+i) BoxIndexer(boxes[i]);
    }

    auto dp = (char*) The_Arena()->alloc(nbytes);
    Gpu::htod_memcpy_async(dp, hp, nbytes);
    pboxes = (BoxIndexer*)dp;
}

inline
void destroy_par_for_boxes (char* hp, char* dp)
{
    // The host-to-device copy in build_par_for_boxes() is asynchronous.  In a
    // NoSync region it may still be pending when the owning ParForInfo is
    // destroyed (e.g., when a temporary FabArray with a unique BoxArray dies).
    // Use stream-ordered frees so that the buffers are not reused too early.
    if (hp) {
        Gpu::freeAsync(The_Pinned_Arena(), hp);
        Gpu::freeAsync(The_Arena(), dp);
    }
}

namespace parfor_mf_detail {
    template <typename F>
    AMREX_GPU_DEVICE
    auto call_f (F const& f, int b, int i, int j, int k, int) noexcept
        -> decltype(f(0,0,0,0))
    {
        f(b,i,j,k);
    }

    template <typename F>
    AMREX_GPU_DEVICE
    auto call_f (F const& f, int b, int i, int j, int k, int ncomp) noexcept
        -> decltype(f(0,0,0,0,0))
    {
        for (int n = 0; n < ncomp; ++n) {
            f(b,i,j,k,n);
        }
    }

    // Row versions for the SYCL CPU device: f is called for the lenx
    // consecutive points (i0, j, k) ... (i0+lenx-1, j, k) of box b. The
    // component loop is outside the row loop; a run-time component loop
    // inside the row loop defeats vectorization (measured 2.6x slower).
    template <typename F>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_row (F const& f, int b, int i0, int lenx, int j, int k, int) noexcept
        -> decltype(f(0,0,0,0))
    {
        for (int i = 0; i < lenx; ++i) {
            f(b,i0+i,j,k);
        }
    }

    template <typename F>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_row (F const& f, int b, int i0, int lenx, int j, int k, int ncomp) noexcept
        -> decltype(f(0,0,0,0,0))
    {
        for (int n = 0; n < ncomp; ++n) {
            for (int i = 0; i < lenx; ++i) {
                f(b,i0+i,j,k,n);
            }
        }
    }
}

// Number of points along x of the box tracked by a BoxIndexer.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int box_indexer_len_x (BoxIndexer const& indexer)
{
#if (AMREX_SPACEDIM == 1)
    return int(indexer.npts);
#else
    return int(indexer.fdm[0].divisor);
#endif
}

template <int MT, FabArrayType MF, typename F>
void
ParallelFor_doit (MF const& mf, IntVect const& nghost, int ncomp, IntVect const&, bool, F const& f)
{
    const auto& index_array = mf.IndexArray();
    const int nboxes = index_array.size();

    if (nboxes == 0) {
        return;
    } else if (nboxes == 1) {
        Box const& b = amrex::grow(mf.box(index_array[0]), nghost);
        amrex::ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            parfor_mf_detail::call_f(f, 0, i, j, k, ncomp);
        });
    } else {
        auto const& parforinfo = mf.getParForInfo(nghost);
        const BoxIndexer* dp_boxes = parforinfo.getBoxes();

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
        if (Gpu::Device::isCpuDevice()) {
            // One x-row per work-item, as in AMReX_TagParallelFor.H. Boxes
            // with fewer rows than the largest one leave their trailing
            // work-items idle.
            const int nrows_per_box = parforinfo.getNRowsPerBox();
            const Long nunits_long = Long(nrows_per_box)*Long(nboxes);
            AMREX_ASSERT(nunits_long < Long(std::numeric_limits<int>::max()));
            const int nunits = int(nunits_long);
            constexpr int G = Gpu::Device::cpu_device_group_size;
            amrex::launch<G>(int((nunits_long+G-1)/G), Gpu::gpuStream(),
                [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
                {
                    const int unit = int(item.get_global_id(0));
                    if (unit >= nunits) { return; }
                    const int ibox = unit / nrows_per_box;
                    const int row = unit - ibox*nrows_per_box;
                    // Local copy: referenced in global memory, the indexer's
                    // fields are reloaded inside the row loop.
                    BoxIndexer const indexer = dp_boxes[ibox];
                    const int lenx = box_indexer_len_x(indexer);
                    const auto icell0 = std::uint64_t(row)*std::uint64_t(lenx);
                    if (icell0 < indexer.numPts()) {
                        auto [i, j, k] = indexer(icell0);
                        parfor_mf_detail::call_f_row(f, ibox, i, lenx, j, k, ncomp);
                    }
                });
            AMREX_GPU_ERROR_CHECK();
            return;
        }
#endif

        auto nblocks_per_box = parforinfo.getNBlocksPerBox(MT);
        AMREX_ASSERT(Long(nblocks_per_box)*Long(nboxes) < Long(std::numeric_limits<int>::max()));
        const int nblocks = nblocks_per_box * nboxes;

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

        amrex::launch_global<MT>
            <<<nblocks, MT, 0, Gpu::gpuStream()>>>
            ([=] AMREX_GPU_DEVICE () noexcept
             {
                 int ibox = int(blockIdx.x) / nblocks_per_box;
                 auto icell = std::uint64_t(blockIdx.x-ibox*nblocks_per_box)*MT + threadIdx.x;

#elif defined(AMREX_USE_SYCL)

        amrex::launch<MT>(nblocks, Gpu::gpuStream(),
             [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
             {
                 int blockIdxx = item.get_group_linear_id();
                 int threadIdxx = item.get_local_linear_id();
                 int ibox = int(blockIdxx) / nblocks_per_box;
                 auto icell = std::uint64_t(blockIdxx-ibox*nblocks_per_box)*MT + threadIdxx;
#endif
                 BoxIndexer const& indexer = dp_boxes[ibox];
                 if (icell < indexer.numPts()) {
                     auto [i, j, k] = indexer(icell);
                     parfor_mf_detail::call_f(f, ibox, i, j, k, ncomp);
                 }
             });
    }
    AMREX_GPU_ERROR_CHECK();
}

template <FabArrayType MF, typename F>
void
ParallelFor_doit (MF const& mf, IntVect const& nghost, int ncomp, IntVect const& ts, bool dynamic, F&& f)
{
#ifdef AMREX_USE_CUDA
    constexpr int MT = 128;
#else
    constexpr int MT = AMREX_GPU_MAX_THREADS;
#endif
    ParallelFor_doit<MT>(mf, nghost, ncomp, ts, dynamic, std::forward<F>(f));
}

template <FabArrayType MF, typename F>
void
ParallelFor_doit (MF const& mf, IntVect const& nghost, IntVect const& ts, bool dynamic, F&& f)
{
#ifdef AMREX_USE_CUDA
    constexpr int MT = 128;
#else
    constexpr int MT = AMREX_GPU_MAX_THREADS;
#endif
    ParallelFor_doit<MT>(mf, nghost, 1, ts, dynamic, std::forward<F>(f));
}

// One launch over the valid points of all boxes whose index equals offset
// modulo stride.
template <int MT, FabArrayType MF, typename F>
void
ParallelForStrided_doit (MF const& mf, IntVect const& stride, IntVect const& offset, F const& f)
{
    const int nboxes = mf.IndexArray().size();
    if (nboxes == 0) { return; }

    auto const& parforinfo = mf.getParForInfo(stride, offset);
    const auto nblocks_per_box = parforinfo.getNBlocksPerBox(MT);
    if (nblocks_per_box == 0) { return; }
    AMREX_ASSERT(Long(nblocks_per_box)*Long(nboxes) < Long(std::numeric_limits<int>::max()));
    const int nblocks = nblocks_per_box * nboxes;
    const BoxIndexer* dp_boxes = parforinfo.getBoxes();

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

    amrex::launch_global<MT>
        <<<nblocks, MT, 0, Gpu::gpuStream()>>>
        ([=] AMREX_GPU_DEVICE () noexcept
         {
             int ibox = int(blockIdx.x) / nblocks_per_box;
             auto icell = std::uint64_t(blockIdx.x-ibox*nblocks_per_box)*MT + threadIdx.x;

#elif defined(AMREX_USE_SYCL)

    amrex::launch<MT>(nblocks, Gpu::gpuStream(),
         [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
         {
             int blockIdxx = item.get_group_linear_id();
             int threadIdxx = item.get_local_linear_id();
             int ibox = int(blockIdxx) / nblocks_per_box;
             auto icell = std::uint64_t(blockIdxx-ibox*nblocks_per_box)*MT + threadIdxx;
#endif
             BoxIndexer const& indexer = dp_boxes[ibox];
             if (icell < indexer.numPts()) {
                 IntVect iv = indexer.intVect(icell);
                 for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                     iv[idim] = indexer.lo[idim] + (iv[idim]-indexer.lo[idim])*stride[idim];
                 }
                 auto const d3 = iv.dim3();
                 f(ibox, d3.x, d3.y, d3.z);
             }
         });
    AMREX_GPU_ERROR_CHECK();
}

// One launch over the red or black points of all boxes. Consecutive threads
// take consecutive points of one row, as in the single-box path.
template <int MT, FabArrayType MF, typename F>
void
ParallelForRedBlack_doit (MF const& mf, int redblack, F const& f)
{
    const int nboxes = mf.IndexArray().size();
    if (nboxes == 0) { return; }

    auto const& parforinfo = mf.getParForInfoRedBlack();
    const auto nblocks_per_box = parforinfo.getNBlocksPerBox(MT);
    if (nblocks_per_box == 0) { return; }
    AMREX_ASSERT(Long(nblocks_per_box)*Long(nboxes) < Long(std::numeric_limits<int>::max()));
    const int nblocks = nblocks_per_box * nboxes;
    const BoxIndexer* dp_boxes = parforinfo.getBoxes();

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

    amrex::launch_global<MT>
        <<<nblocks, MT, 0, Gpu::gpuStream()>>>
        ([=] AMREX_GPU_DEVICE () noexcept
         {
             int ibox = int(blockIdx.x) / nblocks_per_box;
             auto icell = std::uint64_t(blockIdx.x-ibox*nblocks_per_box)*MT + threadIdx.x;

#elif defined(AMREX_USE_SYCL)

    amrex::launch<MT>(nblocks, Gpu::gpuStream(),
         [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
         {
             int blockIdxx = item.get_group_linear_id();
             int threadIdxx = item.get_local_linear_id();
             int ibox = int(blockIdxx) / nblocks_per_box;
             auto icell = std::uint64_t(blockIdxx-ibox*nblocks_per_box)*MT + threadIdxx;
#endif
             // indexers[0]: even i, indexers[1]: odd i. Enumerate the larger
             // lattice, then pick the one with the parity this row needs.
             BoxIndexer const* indexers = dp_boxes + 2*ibox;
             const int big = (indexers[1].numPts() > indexers[0].numPts()) ? 1 : 0;
             if (icell < indexers[big].numPts()) {
                 const IntVect iv = indexers[big].intVect(icell);
                 const int ic = iv[0] - indexers[big].lo[0];
                 auto const d3 = iv.dim3();
                 const int parity = (d3.y + d3.z + redblack) & 1;
                 BoxIndexer const& indexer = indexers[parity];
                 if (ic < box_indexer_len_x(indexer)) {
                     f(ibox, indexer.lo[0] + 2*ic, d3.y, d3.z);
                 }
             }
         });
    AMREX_GPU_ERROR_CHECK();
}

template <FabArrayType MF, typename F>
void
ParallelForStrided_doit (MF const& mf, IntVect const& stride, IntVect const& offset, F const& f)
{
    const auto& index_array = mf.IndexArray();
    if (index_array.size() == 1) {
        Box const& b = strided_box(mf.box(index_array[0]), stride, offset);
        const IntVect lo = b.smallEnd();
        amrex::ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            AMREX_D_TERM(i = lo[0] + (i-lo[0])*stride[0];,
                         j = lo[1] + (j-lo[1])*stride[1];,
                         k = lo[2] + (k-lo[2])*stride[2];)
            f(0, i, j, k);
        });
    } else {
#ifdef AMREX_USE_CUDA
        constexpr int MT = 128;
#else
        constexpr int MT = AMREX_GPU_MAX_THREADS;
#endif
        ParallelForStrided_doit<MT>(mf, stride, offset, f);
    }
}

template <FabArrayType MF, typename F>
void
ParallelForRedBlack_doit (MF const& mf, int redblack, F const& f)
{
    const auto& index_array = mf.IndexArray();
    if (index_array.size() == 1) {
        // Checkerboard: i = 2*ic + ((j+k+redblack)&1)
        Box b = mf.box(index_array[0]);
        const int xlo = b.smallEnd(0);
        const int xhi = b.bigEnd(0);
        b.setSmall(0, amrex::coarsen(xlo,2));
        b.setBig  (0, amrex::coarsen(xhi,2));
        amrex::ParallelFor(b, [=] AMREX_GPU_DEVICE (int ic, int j, int k) noexcept
        {
            const int i = 2*ic + ((j+k+redblack) & 1);
            if (i >= xlo && i <= xhi) {
                f(0, i, j, k);
            }
        });
    } else {
#ifdef AMREX_USE_CUDA
        constexpr int MT = 128;
#else
        constexpr int MT = AMREX_GPU_MAX_THREADS;
#endif
        ParallelForRedBlack_doit<MT>(mf, redblack, f);
    }
}

}
/// \endcond

#endif
#endif
