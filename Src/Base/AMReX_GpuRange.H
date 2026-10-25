#ifndef AMREX_GPU_RANGE_H_
#define AMREX_GPU_RANGE_H_
#include <AMReX_Config.H>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuError.H>
#include <AMReX_Box.H>
#include <AMReX_TypeTraits.H>
#include <concepts>

namespace amrex {

template <std::integral T>
bool isEmpty (T n) noexcept { return n <= 0; }

template<int dim>
AMREX_FORCE_INLINE bool isEmpty (BoxND<dim> const& b) noexcept { return b.isEmpty(); }

namespace Gpu {

/// \cond DOXYGEN_IGNORE
namespace range_detail {

//! integer version
template <std::integral T>
AMREX_GPU_HOST_DEVICE
Long size (T const& b) noexcept { return static_cast<Long>(b); }

template <std::integral T>
AMREX_GPU_HOST_DEVICE
Long at (T const& /*b*/, Long offset) noexcept { return offset; }

//! BoxND version
template<int dim>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE Long size (BoxND<dim> const& b) noexcept
{
    AMREX_IF_ON_DEVICE((return b.numPts();))
    AMREX_IF_ON_HOST((
        amrex::ignore_unused(b);
        return 1;
    ))
}

template<int dim>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE BoxND<dim> at (BoxND<dim> const& b, Long offset) noexcept
{
    AMREX_IF_ON_DEVICE((
        auto iv = b.atOffset(offset);
        return BoxND<dim>(iv,iv,b.ixType());
    ))
    AMREX_IF_ON_HOST((
        amrex::ignore_unused(offset);
        return b;
    ))
}

template <typename T>
struct range_impl
{
    AMREX_GPU_HOST_DEVICE
    explicit range_impl (T const& b) noexcept : m_b(b), m_n(range_detail::size(b)) {}

#ifdef AMREX_USE_SYCL
    range_impl (T const& b, Long gid, Long grange) noexcept
        : m_b(b), m_n(range_detail::size(b)), m_gid(gid), m_grange(grange) {}
#endif

    struct iterator
    {
        AMREX_GPU_HOST_DEVICE
        iterator (T const& b, Long i, Long s) noexcept : mi_b(&b), mi_i(i), mi_s(s)  {}

        AMREX_GPU_HOST_DEVICE
        void operator++ () noexcept { mi_i += mi_s; }

        AMREX_GPU_HOST_DEVICE
        bool operator!= (iterator const& rhs) const noexcept { return mi_i < rhs.mi_i; }

        AMREX_GPU_HOST_DEVICE
        T operator* () const noexcept { return range_detail::at(*mi_b,mi_i); }

    private:
        T const* mi_b;
        Long mi_i;
        Long mi_s;
    };

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    iterator begin () const noexcept {
#if defined (__SYCL_DEVICE_ONLY__)
        return iterator(m_b, m_gid, m_grange);
#elif defined(AMREX_USE_SYCL) && defined(__ACPP__)
        AMREX_IF_ON_DEVICE((
            return iterator(m_b, m_gid, m_grange);
        ))
        AMREX_IF_ON_HOST((
            return iterator(m_b,0,1);
        ))
#else
        AMREX_IF_ON_DEVICE((
            return iterator(m_b, blockDim.x*blockIdx.x+threadIdx.x, blockDim.x*gridDim.x);
        ))
        AMREX_IF_ON_HOST((
            return iterator(m_b,0,1);
        ))
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    iterator end () const noexcept { return iterator(m_b,m_n,0); }

private:
    T m_b;
    Long m_n;
#ifdef AMREX_USE_SYCL
    Long m_gid;
    Long m_grange;
#endif
};

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)

template <typename T> inline constexpr bool is_box_v = false;
template <int dim> inline constexpr bool is_box_v<BoxND<dim>> = true;

//! Number of units of a RowRange: the x-rows of a box, the elements of an
//! integer range.
template <int dim>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Long num_rows (BoxND<dim> const& b) noexcept
{
    return b.isEmpty() ? Long(0) : b.numPts() / b.length(0);
}

template <std::integral T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Long num_rows (T const& n) noexcept { return static_cast<Long>(n); }

//! Grid-stride range over the x-rows of a box: work-item gid gets rows gid,
//! gid+grange, ..., each as a box of one row. Used by the range launches on
//! a SYCL CPU device, where one row per work-item lets the kernel's own
//! i-loop run contiguously instead of one cell per work-item.
template <int dim>
struct row_range_impl
{
    row_range_impl (BoxND<dim> const& b, Long gid, Long grange) noexcept
        : m_b(b), m_len0(b.isEmpty() ? 0 : b.length(0)), m_nrows(num_rows(b)),
          m_gid(gid), m_grange(grange) {}

    struct iterator
    {
        AMREX_GPU_HOST_DEVICE
        iterator (row_range_impl const& r, Long row) noexcept : mi_r(&r), mi_row(row) {}

        AMREX_GPU_HOST_DEVICE
        void operator++ () noexcept { mi_row += mi_r->m_grange; }

        AMREX_GPU_HOST_DEVICE
        bool operator!= (iterator const& rhs) const noexcept { return mi_row < rhs.mi_row; }

        AMREX_GPU_HOST_DEVICE
        BoxND<dim> operator* () const noexcept
        {
            auto const lo = mi_r->m_b.atOffset(mi_row*mi_r->m_len0);
            auto hi = lo;
            hi[0] += mi_r->m_len0 - 1;
            return BoxND<dim>(lo, hi, mi_r->m_b.ixType());
        }

    private:
        row_range_impl const* mi_r;
        Long mi_row;
    };

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    iterator begin () const noexcept { return iterator(*this, m_gid); }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    iterator end () const noexcept { return iterator(*this, m_nrows); }

private:
    BoxND<dim> m_b;
    int m_len0;
    Long m_nrows;
    Long m_gid;
    Long m_grange;
};

#endif
}
/// \endcond

#ifdef AMREX_USE_SYCL
template <typename T>
range_detail::range_impl<T> Range (T const& b, Long gid, Long grange) noexcept {
    return range_detail::range_impl<T>(b,gid,grange);
}
#endif

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)
//! Row version of Range(b, gid, grange) for the SYCL CPU device; an integer
//! range is the same as Range.
template <int dim>
range_detail::row_range_impl<dim> RowRange (BoxND<dim> const& b, Long gid, Long grange) noexcept {
    return range_detail::row_range_impl<dim>(b,gid,grange);
}

template <std::integral T>
range_detail::range_impl<T> RowRange (T const& n, Long gid, Long grange) noexcept {
    return range_detail::range_impl<T>(n,gid,grange);
}
#endif

template <typename T>
AMREX_GPU_HOST_DEVICE
range_detail::range_impl<T> Range (T const& b) noexcept { return range_detail::range_impl<T>(b); }

}}

#endif
