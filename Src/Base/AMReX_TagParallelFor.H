#ifndef AMREX_TAG_PARALLELFOR_H_
#define AMREX_TAG_PARALLELFOR_H_
#include <AMReX_Config.H>

#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Vector.H>
#include <limits>
#include <utility>

namespace amrex {

template <class T>
struct Array4PairTag {
    Array4<T      > dfab;
    Array4<T const> sfab;
    Box dbox;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T0, class T1=T0>
struct Array4CopyTag {
    Array4<T0      > dfab;
    int              dindex;
    Array4<T1 const> sfab;
    Box dbox;
    Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T0, class T1=T0>
struct Array4MaskCopyTag {
    Array4<T0      > dfab;
    Array4<T1 const> sfab;
    Array4<int     > mask;
    Box dbox;
    Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4Tag {
    Array4<T> dfab;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box box () const noexcept { return Box(dfab); }
};

template <class T>
struct Array4BoxTag {
    Array4<T> dfab;
    Box       dbox;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4BoxValTag {
    Array4<T> dfab;
    Box       dbox;
    T          val;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4BoxOrientationTag {
    Array4<T> fab;
    Box bx;
    Orientation face;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box() const noexcept { return bx; }
};

template <class T>
struct Array4BoxOffsetTag {
    Array4<T> fab;
    Box bx;
    Dim3 offset;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box() const noexcept { return bx; }
};

template <class T>
struct VectorTag {
    T* p;
    Long m_size;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Long size () const noexcept { return m_size; }
};

template <class T>
struct CommRecvBufTag { // for unpacking recv buffer
    Array4<T> dfab;
    std::ptrdiff_t poff;
    Box bx;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return bx; }
};

template <class T>
struct CommSendBufTag { // for packing send buffer
    Array4<T const> sfab;
    std::ptrdiff_t poff;
    Box bx;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return bx; }
};

/// \cond DOXYGEN_IGNORE
namespace detail {

    template <typename T>
    requires (std::same_as<std::decay_t<decltype(std::declval<T>().box())>, Box>)
    Long get_tag_size (T const& tag) noexcept
    {
        AMREX_ASSERT(tag.box().numPts() < Long(std::numeric_limits<int>::max()));
        return static_cast<int>(tag.box().numPts());
    }

    template <typename T>
    requires (std::is_integral_v<std::decay_t<decltype(std::declval<T>().size())> >)
    Long get_tag_size (T const& tag) noexcept
    {
        AMREX_ASSERT(tag.size() < Long(std::numeric_limits<int>::max()));
        return tag.size();
    }

    template <typename T>
    requires (std::same_as<std::decay_t<decltype(std::declval<T>().box())>, Box>)
    constexpr bool is_box_tag (T const&) { return true; }

    template <typename T>
    requires (std::is_integral_v<std::decay_t<decltype(std::declval<T>().size())> >)
    constexpr bool is_box_tag (T const&) { return false; }

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
    // Cells (box tags) or elements (vector tags) per work unit on a SYCL
    // CPU device.
    constexpr int tag_cpu_chunk_size = 256;

    // x-rows per work unit of a box tag with lengths len_x, len_y on a SYCL
    // CPU device: about tag_cpu_chunk_size cells, at least one row and at
    // most one k-plane, so that a unit is a box. A box one cell wide in x
    // (e.g. an x-face) then pays the per-unit tag lookup once per unit
    // instead of once per cell.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int tag_cpu_rows_per_unit (int len_x, int len_y) noexcept
    {
        return amrex::min(len_y, amrex::max(1, tag_cpu_chunk_size/len_x));
    }

    // Cells of unit `unit` of a box tag with box bx on a SYCL CPU device:
    // rows [j0,j1) of one k-plane.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box tag_cpu_unit_box (int unit, Box const& bx) noexcept
    {
        const auto len = amrex::length(bx);
        const auto lo  = amrex::lbound(bx);
        const int rows = tag_cpu_rows_per_unit(len.x, len.y);
        const int nju = (len.y + rows - 1) / rows;
        const int k = unit / nju;
        const int j0 = (unit - k*nju) * rows;
        const int j1 = amrex::min(j0+rows, len.y);
        return Box(IntVect(AMREX_D_DECL(lo.x, lo.y+j0, lo.z+k)),
                   IntVect(AMREX_D_DECL(lo.x+len.x-1, lo.y+j1-1, lo.z+k)),
                   bx.ixType());
    }
#endif

#ifdef AMREX_USE_GPU
    // A tag is split into work units. On GPUs a unit is one warp of cells,
    // one cell per thread. On a SYCL CPU device (AdaptiveCpp's OpenMP
    // backend) a work-item is one scalar loop iteration, so a unit is the
    // box tag_cpu_unit_box of a box tag, or tag_cpu_chunk_size consecutive
    // elements of a vector tag, and the work-item loops over the unit's
    // cells.
    template <typename T>
    Long get_tag_nunits (T const& tag) noexcept
    {
#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
        if (Gpu::Device::isCpuDevice()) {
            if constexpr (is_box_tag(T{})) {
                auto const len = amrex::length(tag.box());
                if (len.x <= 0 || len.y <= 0) { return Long(0); }
                Long const rows = tag_cpu_rows_per_unit(len.x, len.y);
                return Long(len.z) * ((len.y + rows - 1) / rows);
            } else {
                return (get_tag_size(tag) + tag_cpu_chunk_size - 1) / tag_cpu_chunk_size;
            }
        }
#endif
        return (get_tag_size(tag) + Gpu::Device::warp_size-1) / Gpu::Device::warp_size;
    }
#endif
}
/// \endcond

template <class TagType>
struct TagVector {

    char* h_buffer = nullptr;
    char* d_buffer = nullptr;
    TagType* d_tags =  nullptr;
    int* d_nunits = nullptr;   // prefix sum of work units per tag, ntags+1 entries
    int ntags = 0;
    int ntotunits = 0;
    int nblocks = 0;
    bool defined = false;
    static constexpr int nthreads = 256;

    TagVector () = default;

    TagVector (Vector<TagType> const& tags) {
        define(tags);
    }

    ~TagVector () {
        if (defined) {
            undefine();
        }
    }

    TagVector (const TagVector& other) = delete;
    TagVector& operator= (const TagVector& other) = delete;
    TagVector (TagVector&& other) noexcept
        : h_buffer{other.h_buffer},
          d_buffer{other.d_buffer},
          d_tags{other.d_tags},
          d_nunits{other.d_nunits},
          ntags{other.ntags},
          ntotunits{other.ntotunits},
          nblocks{other.nblocks},
          defined{other.defined}
    {
        other.h_buffer = nullptr;
        other.d_buffer = nullptr;
        other.d_tags = nullptr;
        other.d_nunits = nullptr;
        other.ntags = 0;
        other.ntotunits = 0;
        other.nblocks = 0;
        other.defined = false;
    }
    TagVector& operator= (TagVector&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        undefine();
        h_buffer = other.h_buffer;
        other.h_buffer = nullptr;
        d_buffer = other.d_buffer;
        other.d_buffer = nullptr;
        d_tags = other.d_tags;
        other.d_tags = nullptr;
        d_nunits = other.d_nunits;
        other.d_nunits = nullptr;
        ntags = other.ntags;
        other.ntags = 0;
        ntotunits = other.ntotunits;
        other.ntotunits = 0;
        nblocks = other.nblocks;
        other.nblocks = 0;
        defined = other.defined;
        other.defined = false;
        return *this;
    }

    [[nodiscard]] bool is_defined () const { return defined; }

    void define (Vector<TagType> const& tags) {
        if (defined) {
            undefine();
        }

        ntags = tags.size();
        if (ntags == 0) {
            defined = true;
            return;
        }

#ifdef AMREX_USE_GPU
        Long l_ntotunits = 0;
        ntotunits = 0;
        Vector<int> nunits;
        nunits.reserve(ntags+1);
        for (int i = 0; i < ntags; ++i)
        {
            auto& tag = tags[i];
            nunits.push_back(ntotunits);
            auto nu = detail::get_tag_nunits(tag);
            l_ntotunits += nu;
            ntotunits += static_cast<int>(nu);
        }
        nunits.push_back(ntotunits);

        std::size_t sizeof_tags = ntags*sizeof(TagType);
        std::size_t offset_nunits = Arena::align(sizeof_tags);
        std::size_t sizeof_nunits = (ntags+1)*sizeof(int);
        std::size_t total_buf_size = offset_nunits + sizeof_nunits;

        h_buffer = (char*)The_Pinned_Arena()->alloc(total_buf_size);
        d_buffer = (char*)The_Arena()->alloc(total_buf_size);

        std::memcpy(h_buffer, tags.data(), sizeof_tags);
        std::memcpy(h_buffer+offset_nunits, nunits.data(), sizeof_nunits);
        Gpu::htod_memcpy_async(d_buffer, h_buffer, total_buf_size);

        d_tags = reinterpret_cast<TagType*>(d_buffer);
        d_nunits = reinterpret_cast<int*>(d_buffer+offset_nunits);

        // Units per block: nthreads/warp_size on GPUs (one warp per unit).
        // On a SYCL CPU device each work-item handles a whole unit and the
        // block is the small cpu_device_group_size (see AMReX_GpuDevice.H).
        int nunits_per_block = nthreads/Gpu::Device::warp_size;
#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
        if (Gpu::Device::isCpuDevice()) { nunits_per_block = Gpu::Device::cpu_device_group_size; }
#endif
        nblocks = (ntotunits + nunits_per_block-1) / nunits_per_block;

        defined = true;

        amrex::ignore_unused(l_ntotunits);
        AMREX_ALWAYS_ASSERT(l_ntotunits+nunits_per_block-1 < Long(std::numeric_limits<int>::max()));
#else
        std::size_t sizeof_tags = ntags*sizeof(TagType);
        h_buffer = (char*)The_Pinned_Arena()->alloc(sizeof_tags);

        std::memcpy(h_buffer, tags.data(), sizeof_tags);

        d_tags = reinterpret_cast<TagType*>(h_buffer);

        defined = true;
#endif
    }

    void undefine () {
        if (defined) {
            Gpu::streamSynchronize();
            The_Pinned_Arena()->free(h_buffer);
            The_Arena()->free(d_buffer);
            h_buffer = nullptr;
            d_buffer = nullptr;
            d_tags = nullptr;
            d_nunits = nullptr;
            ntags = 0;
            ntotunits = 0;
            nblocks = 0;
            defined = false;
        }
    }
};

/// \cond DOXYGEN_IGNORE
namespace detail {

#ifdef AMREX_USE_GPU

template <typename T, typename F>
requires (std::same_as<std::decay_t<decltype(std::declval<T>().box())>, Box>)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tagparfor_call_f (int icell, T const& tag, F&& f) noexcept
{
    int ncells = tag.box().numPts();
    const auto len = amrex::length(tag.box());
    const auto lo  = amrex::lbound(tag.box());
    int k =  icell /   (len.x*len.y);
    int j = (icell - k*(len.x*len.y)) /   len.x;
    int i = (icell - k*(len.x*len.y)) - j*len.x;
    i += lo.x;
    j += lo.y;
    k += lo.z;
    f(icell, ncells, i, j, k, tag);
}

template <typename T, typename F>
requires (std::is_integral_v<std::decay_t<decltype(std::declval<T>().size())> >)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tagparfor_call_f (int i, T const& tag, F&& f) noexcept
{
    int N = tag.size();
    f(i, N, tag);
}

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
// SYCL CPU device: one work-item handles a whole unit (see get_tag_nunits).
// Box tag: unit `unit` is the box tag_cpu_unit_box (x-rows of one k-plane);
// f is called for each cell with the same (icell, ncells, i, j, k, tag)
// arguments as on GPUs.
template <typename T, typename F>
requires (std::same_as<std::decay_t<decltype(std::declval<T>().box())>, Box>)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tagparfor_call_unit (int unit, T const& tag, F&& f) noexcept
{
    int ncells = tag.box().numPts();
    const auto len = amrex::length(tag.box());
    const auto lo  = amrex::lbound(tag.box());
    const auto ub  = tag_cpu_unit_box(unit, tag.box());
    const auto ulo = amrex::lbound(ub);
    const auto uhi = amrex::ubound(ub);
    const int k = ulo.z;
    for (int j = ulo.y; j <= uhi.y; ++j) {
        int icell = ((k-lo.z)*len.y + (j-lo.y))*len.x;
        for (int i = 0; i < len.x; ++i) {
            f(icell+i, ncells, lo.x+i, j, k, tag);
        }
    }
}

// Box tag with components: the component loop is outside the row loop, so
// the inner loop over i has no nested loop of run-time trip count (that
// blocked strength reduction and vectorization; measured 2.6x slower).
template <typename T, typename F>
requires (std::same_as<std::decay_t<decltype(std::declval<T>().box())>, Box>)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tagparfor_call_unit (int unit, int ncomp, T const& tag, F&& f) noexcept
{
    int ncells = tag.box().numPts();
    const auto len = amrex::length(tag.box());
    const auto lo  = amrex::lbound(tag.box());
    const auto ub  = tag_cpu_unit_box(unit, tag.box());
    const auto ulo = amrex::lbound(ub);
    const auto uhi = amrex::ubound(ub);
    const int k = ulo.z;
    for (int n = 0; n < ncomp; ++n) {
        for (int j = ulo.y; j <= uhi.y; ++j) {
            int icell = ((k-lo.z)*len.y + (j-lo.y))*len.x;
            for (int i = 0; i < len.x; ++i) {
                f(icell+i, ncells, lo.x+i, j, k, n, tag);
            }
        }
    }
}

// Vector tag: unit `chunk` is tag_cpu_chunk_size consecutive elements.
template <typename T, typename F>
requires (std::is_integral_v<std::decay_t<decltype(std::declval<T>().size())> >)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tagparfor_call_unit (int chunk, T const& tag, F&& f) noexcept
{
    int N = tag.size();
    int ibeg = chunk*tag_cpu_chunk_size;
    int iend = amrex::min(ibeg+tag_cpu_chunk_size, N);
    for (int i = ibeg; i < iend; ++i) {
        f(i, N, tag);
    }
}
#endif

template <class TagType, class F>
void
ParallelFor_doit (TagVector<TagType> const& tv, F const& f)
{
    AMREX_ALWAYS_ASSERT(tv.is_defined());

    if (tv.ntags == 0 || tv.ntotunits == 0) { return; }

    const auto d_tags = tv.d_tags;
    const auto d_nunits = tv.d_nunits;
    const auto ntags = tv.ntags;
    const auto ntotunits = tv.ntotunits;
    constexpr auto nthreads = TagVector<TagType>::nthreads;

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
    if (Gpu::Device::isCpuDevice()) {
        // One unit (rows or chunk) per work-item in small groups; the
        // TagVector was defined with the matching unit layout.
        amrex::launch<Gpu::Device::cpu_device_group_size>(tv.nblocks, Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
        {
            int unit = int(item.get_global_id(0));
            if (unit >= ntotunits) { return; }
            int tag_id = amrex::bisect(d_nunits, 0, ntags, unit);
            // Local copy of the tag: with the tag referenced in global memory
            // the compiler reloads the Array4 strides inside the row loop
            // (measured 1.6x slower).
            TagType const tag = d_tags[tag_id];
            tagparfor_call_unit(unit - d_nunits[tag_id], tag, f);
        });
        return;
    }
#endif

    amrex::launch<nthreads>(tv.nblocks, Gpu::gpuStream(),
#ifdef AMREX_USE_SYCL
    [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
    AMREX_SYCL_REQD_WORK_GROUP_SIZE(nthreads)
    AMREX_SYCL_REQD_SUB_GROUP_SIZE(Gpu::Device::warp_size)
#else
    [=] AMREX_GPU_DEVICE () noexcept
#endif
    {
#ifdef AMREX_USE_SYCL
        std::size_t g_tid = item.get_global_id(0);
#else
        auto g_tid = std::size_t(blockDim.x)*blockIdx.x + threadIdx.x;
#endif
        auto g_wid = int(g_tid / Gpu::Device::warp_size);
        if (g_wid >= ntotunits) { return; }

        int tag_id = amrex::bisect(d_nunits, 0, ntags, g_wid);

        int b_wid = g_wid - d_nunits[tag_id]; // b_wid'th warp on this box
#ifdef AMREX_USE_SYCL
        int lane = item.get_local_id(0) % Gpu::Device::warp_size;
#else
        int lane = threadIdx.x % Gpu::Device::warp_size;
#endif
        int icell = b_wid*Gpu::Device::warp_size + lane;

        tagparfor_call_f(icell, d_tags[tag_id], f);
    });
}

// Box tags with ncomp components: f(icell, ncells, i, j, k, n, tag) is
// called once per cell and component. On GPUs each thread loops over the
// components of its cell; on a SYCL CPU device the component loop is
// outside the row loop (see tagparfor_call_unit).
template <class TagType, class F>
requires (std::same_as<std::decay_t<decltype(std::declval<TagType>().box())>, Box>)
void
ParallelFor_doit (TagVector<TagType> const& tv, int ncomp, F const& f)
{
#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
    if (Gpu::Device::isCpuDevice()) {
        AMREX_ALWAYS_ASSERT(tv.is_defined());
        if (tv.ntags == 0 || tv.ntotunits == 0) { return; }

        const auto d_tags = tv.d_tags;
        const auto d_nunits = tv.d_nunits;
        const auto ntags = tv.ntags;
        const auto ntotunits = tv.ntotunits;

        amrex::launch<Gpu::Device::cpu_device_group_size>(tv.nblocks, Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
        {
            int unit = int(item.get_global_id(0));
            if (unit >= ntotunits) { return; }
            int tag_id = amrex::bisect(d_nunits, 0, ntags, unit);
            TagType const tag = d_tags[tag_id];
            tagparfor_call_unit(unit - d_nunits[tag_id], ncomp, tag, f);
        });
        return;
    }
#endif
    ParallelFor_doit(tv,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, int i, int j, int k, TagType const& tag) noexcept
    {
        for (int n = 0; n < ncomp; ++n) {
            f(icell, ncells, i, j, k, n, tag);
        }
    });
}

#else // ifdef AMREX_USE_GPU

template <class TagType, class F>
void
ParallelFor_doit (TagVector<TagType> const& tv, F const& f)
{
    // Note: this CPU version may not have optimal performance:
    // The loop over ncomp is the innermost instead of the outermost
    // There is no load-balancing or splitting of tags
    AMREX_ALWAYS_ASSERT(tv.is_defined());

    constexpr bool tag_type = is_box_tag(TagType{});

    if (tv.ntags == 0) { return; }

    const auto d_tags = tv.d_tags;
    const auto ntags = tv.ntags;

#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
    for (int itag = 0; itag < ntags; ++itag) {

        const auto& t = d_tags[itag];

        if constexpr (tag_type) {
            const auto lo = amrex::lbound(t.box());
            const auto hi = amrex::ubound(t.box());

            for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {
                        f(0, 1, i, j, k, t);
                    }
                }
            }
        } else {
            const auto size = t.size();

            AMREX_PRAGMA_SIMD
            for (int i = 0; i < size; ++i) {
                f(i, size, t);
            }
        }
    }
}

#endif

}
/// \endcond

template <class TagType, class F>
requires (std::same_as<std::decay_t<decltype(std::declval<TagType>().box())>, Box>)
void ParallelFor (TagVector<TagType> const& tv, int ncomp, F const& f)
{
#ifdef AMREX_USE_GPU
    detail::ParallelFor_doit(tv, ncomp,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, int i, int j, int k, int n, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                f(i,j,k,n,tag);
            }
        });
#else
    detail::ParallelFor_doit(tv,
        [=] (int icell, int ncells, int i, int j, int k, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                for (int n = 0; n < ncomp; ++n) {
                    f(i,j,k,n,tag);
                }
            }
        });
#endif
}

template <class TagType, class F>
requires (std::same_as<std::decay_t<decltype(std::declval<TagType>().box())>, Box>)
void ParallelFor (TagVector<TagType> const& tv, F const& f)
{
    detail::ParallelFor_doit(tv,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, int i, int j, int k, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                f(i,j,k,tag);
            }
        });
}

template <class TagType, class F>
requires (std::is_integral_v<std::decay_t<decltype(std::declval<TagType>().size())> >)
void ParallelFor (TagVector<TagType> const& tv, F const& f)
{
    detail::ParallelFor_doit(tv,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                f(icell,tag);
            }
        });
}

template <class TagType, class F>
requires (std::same_as<std::decay_t<decltype(std::declval<TagType>().box())>, Box>)
void ParallelFor (Vector<TagType> const& tags, int ncomp, F && f)
{
    TagVector<TagType> tv{tags};
    ParallelFor(tv, ncomp, std::forward<F>(f));
}

template <class TagType, class F>
requires (std::same_as<std::decay_t<decltype(std::declval<TagType>().box())>, Box>)
void ParallelFor (Vector<TagType> const& tags, F && f)
{
    TagVector<TagType> tv{tags};
    ParallelFor(tv, std::forward<F>(f));
}

template <class TagType, class F>
requires (std::is_integral_v<std::decay_t<decltype(std::declval<TagType>().size())> >)
void ParallelFor (Vector<TagType> const& tags, F && f)
{
    TagVector<TagType> tv{tags};
    ParallelFor(tv, std::forward<F>(f));
}

}

#endif
