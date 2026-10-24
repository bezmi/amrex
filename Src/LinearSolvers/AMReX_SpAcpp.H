#ifndef AMREX_SP_ACPP_H_
#define AMREX_SP_ACPP_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_SpAcpp.H
 *
 * Portable SYCL kernels for the sparse operations on AdaptiveCpp GPU devices:
 * CSR row sorting, transpose, SpMV and SpGEMM.
 *
 * AdaptiveCpp builds do not use vendor sparse libraries: oneMath, which
 * replaces oneMKL there, provides only SpMV, and only through cuSPARSE or
 * rocSPARSE. On the AdaptiveCpp CPU device the callers use AMReX's host code
 * instead of the sort and SpGEMM kernels here, because AdaptiveCpp's OpenMP
 * backend fails to JIT-compile kernels that combine work-group barriers with
 * data-dependent loops. The SpMV and transpose kernels without barriers run on
 * both devices.
 */

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_Scan.H>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace amrex {

template <typename T, typename Index> struct CsrView;
template <typename T, template <typename> class V, typename Index> struct CSR;

namespace detail::sp_acpp {

//! Work-group size of the kernels that use local memory.
constexpr int wg_size = 256;

//! True if the device is AdaptiveCpp's CPU device.
inline bool on_cpu_device ()
{
    return Gpu::Device::syclDevice().is_cpu();
}

//! Aborts if a kernel's local memory request exceeds what the device offers.
inline void check_local_memory (std::size_t bytes)
{
    static std::size_t const avail =
        Gpu::Device::syclDevice().get_info<sycl::info::device::local_mem_size>();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes <= avail,
        "AMReX_SpAcpp.H: kernel needs more local memory than the device has");
}

//! Number of work-groups of `wg_size` work-items that cover n items.
inline int num_blocks (Long n)
{
    AMREX_ALWAYS_ASSERT(n <= Long(std::numeric_limits<int>::max()) * wg_size);
    return int((n + wg_size - 1) / wg_size);
}

// ---------------------------------------------------------------- atomics

// Atomics on local or global memory. AS names the memory the pointer refers
// to: atomics declared on local memory with work-group scope compile to the
// GPU's shared-memory atomic instructions; generic-address device-scope
// atomics on the same data run about twice as slow.
using address_space = sycl::access::address_space;

//! Memory scope matching the memory: work-group for local memory.
template <address_space AS>
constexpr sycl::memory_scope scope_of ()
{
    return (AS == address_space::local_space) ? sycl::memory_scope::work_group
                                              : sycl::memory_scope::device;
}

template <address_space AS, typename U>
using atomic_ref_t = sycl::atomic_ref<U, sycl::memory_order::relaxed, scope_of<AS>(), AS>;

//! p += v for real and complex values (complex through its two parts).
template <address_space AS, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void atomic_add (T* p, T v)
{
    if constexpr (std::is_arithmetic_v<T>) {
        atomic_ref_t<AS,T>(*p).fetch_add(v);
    } else {
        using U = typename T::value_type;
        auto* q = reinterpret_cast<U*>(p);
        atomic_add<AS>(q, v.real());
        atomic_add<AS>(q+1, v.imag());
    }
}

//! Compare-and-swap; returns the value found at p.
template <address_space AS, typename I>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
I atomic_cas (I* p, I expected, I desired)
{
    atomic_ref_t<AS,I>(*p).compare_exchange_strong(expected, desired);
    return expected;
}

//! ++*p; returns the old value.
template <address_space AS, typename I>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
I atomic_inc (I* p)
{
    return atomic_ref_t<AS,I>(*p).fetch_add(I(1));
}

// ---------------------------------------------------------------- local memory

/**
 * Carves typed arrays out of a work-group's dynamic local memory, in order,
 * each aligned to its type. The same carving function called with
 * LocalBytes gives the number of bytes to request at launch, so the byte
 * count and the device pointers cannot disagree.
 */
struct LocalArena
{
    char* p;

    template <typename U>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    U* take (std::size_t n) noexcept
    {
        auto a = reinterpret_cast<std::uintptr_t>(p);
        a = (a + alignof(U) - 1) / alignof(U) * alignof(U);
        p = reinterpret_cast<char*>(a + n*sizeof(U));
        return reinterpret_cast<U*>(a);
    }
};

//! Byte counter with LocalArena's interface; take() returns nullptr.
struct LocalBytes
{
    std::size_t n = 0;

    template <typename U>
    U* take (std::size_t count) noexcept
    {
        n = (n + alignof(U) - 1) / alignof(U) * alignof(U) + count*sizeof(U);
        return nullptr;
    }
};

// ---------------------------------------------------------------- row sort

/**
 * Sorts the rows of a CSR matrix that have more than `min_len` entries and
 * are not already sorted; GPU devices only. A team of 32 work-items ranks
 * each row: rows of at most 128 entries in local memory, longer rows through
 * temporary global arrays. An entry's rank is the number of entries with a
 * smaller column index, or an equal one earlier in the row, so the sort is
 * stable. Synchronous.
 *
 * \param nrows   number of rows
 * \param row     row offsets (nrows+1 entries)
 * \param col     column indices, sorted on return
 * \param val     values, permuted with the columns
 * \param min_len rows with at most this many entries are left to the caller
 */
template <typename T, typename Index>
void sort_rows (Long nrows, Index const* row, Index* col, T* val, int min_len)
{
    if (nrows <= 0) { return; }

    constexpr int team = 32;
    constexpr int teams = wg_size / team;
    constexpr int cap = 128;

    // Count the rows this kernel must sort, and those too long for local memory.
    Gpu::Buffer<int> counts({0, 0});
    auto* pcounts = counts.data();
    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (Long r) noexcept
    {
        Long const b = row[r];
        Long const e = row[r+1];
        if (e - b <= min_len) { return; }
        for (Long i = b+1; i < e; ++i) {
            if (col[i-1] > col[i]) {
                Gpu::Atomic::AddNoRet(pcounts, 1);
                if (e - b > cap) { Gpu::Atomic::AddNoRet(pcounts+1, 1); }
                return;
            }
        }
    });
    auto const* hcounts = counts.copyToHost();
    if (hcounts[0] == 0) { return; }

    // Temporaries, needed for the long rows only.
    Long nnz_tmp = 0;
    if (hcounts[1] > 0) {
        Index nnz_h = 0;
        Gpu::dtoh_memcpy_async(&nnz_h, row + nrows, sizeof(Index));
        Gpu::streamSynchronize();
        nnz_tmp = nnz_h;
    }
    Gpu::DeviceVector<Index> col_tmp(nnz_tmp);
    Gpu::DeviceVector<T> val_tmp(nnz_tmp);
    auto* pct = col_tmp.data();
    auto* pvt = val_tmp.data();

    auto carve = [] (auto& arena) {
        struct { T* val; Index* col; } s;
        s.val = arena.template take<T>(std::size_t(teams)*cap);
        s.col = arena.template take<Index>(std::size_t(teams)*cap);
        return s;
    };
    LocalBytes bytes;
    carve(bytes);
    check_local_memory(bytes.n);

    amrex::launch<wg_size>(num_blocks(nrows*team), bytes.n, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
    {
        LocalArena arena{static_cast<char*>(h.local)};
        auto const local = carve(arena);
        int const lid = int(h.threadIdx());
        int const t = lid / team;
        int const lane = lid % team;
        Long const r = Long(h.blockIdx())*teams + t;
        Long b = 0, e = 0;
        if (r < nrows) { b = row[r]; e = row[r+1]; }
        auto const n = int(e - b);
        bool const active = n > min_len;
        bool const in_local = n <= cap;
        // Keys and values read from kc/kv, ranks written to oc/ov.
        Index* kc = in_local ? local.col + t*cap : col + b;
        T* kv = in_local ? local.val + t*cap : val + b;
        Index* oc = in_local ? col + b : pct + b;
        T* ov = in_local ? val + b : pvt + b;
        if (active && in_local) {
            for (int j = lane; j < n; j += team) { kc[j] = col[b+j]; kv[j] = val[b+j]; }
        }
        sycl::group_barrier(h.item->get_group());
        if (active) {
            for (int j = lane; j < n; j += team) {
                Index const key = kc[j];
                int rank = 0;
                for (int m = 0; m < n; ++m) {
                    Index const km = kc[m];
                    rank += (km < key || (km == key && m < j)) ? 1 : 0;
                }
                oc[rank] = key;
                ov[rank] = kv[j];
            }
        }
        sycl::group_barrier(h.item->get_group());
        if (active && !in_local) {
            for (int j = lane; j < n; j += team) { col[b+j] = pct[b+j]; val[b+j] = pvt[b+j]; }
        }
    });
    Gpu::streamSynchronize();
}

// ---------------------------------------------------------------- transpose

/**
 * CSR transpose, csrt = transpose of csr; both views must be allocated for
 * the transposed sizes (csrt has csr's column count as its row count and the
 * same nnz). Counts the entries per column, scans the counts into the row
 * offsets of csrt, scatters the entries with an atomic cursor per row of csrt,
 * and sorts the rows of csrt. Synchronous.
 */
template <typename T, typename IO, typename II>
void transpose (CsrView<T,IO> const& csrt, CsrView<T const,II> const& csr)
{
    Long const nrows = csr.nrows;
    Long const ncols = csrt.nrows;
    Long const nnz = csr.nnz;
    AMREX_ALWAYS_ASSERT(nnz <= Long(std::numeric_limits<IO>::max()));

    auto const* row = csr.row_offset;
    auto const* col = csr.col_index;
    auto const* val = csr.mat;
    auto* trow = csrt.row_offset;
    auto* tcol = csrt.col_index;
    auto* tval = csrt.mat;

    Gpu::DeviceVector<IO> cnt(ncols);
    auto* pcnt = cnt.data();
    ParallelFor(ncols, [=] AMREX_GPU_DEVICE (Long i) noexcept { pcnt[i] = 0; });
    ParallelFor(nnz, [=] AMREX_GPU_DEVICE (Long j) noexcept
    {
        Gpu::Atomic::AddNoRet(pcnt + col[j], IO(1));
    });
    Scan::ExclusiveSum(ncols, pcnt, trow, Scan::noRetSum);
    ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept { trow[ncols] = IO(nnz); });

    // The counts become the cursors of the scatter.
    ParallelFor(ncols, [=] AMREX_GPU_DEVICE (Long i) noexcept { pcnt[i] = trow[i]; });
    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (Long i) noexcept
    {
        for (Long p = row[i]; p < row[i+1]; ++p) {
            IO const pos = Gpu::Atomic::Add(pcnt + col[p], IO(1));
            tcol[pos] = IO(i);
            tval[pos] = val[p];
        }
    });

    if (on_cpu_device()) {
        // One work-item per row; the CPU device cannot run sort_rows.
        ParallelFor(ncols, [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            Long const b = trow[i];
            Long const e = trow[i+1];
            for (Long k = b+1; k < e; ++k) {
                IO const key = tcol[k];
                T const v = tval[k];
                Long m = k-1;
                while (m >= b && tcol[m] > key) {
                    tcol[m+1] = tcol[m];
                    tval[m+1] = tval[m];
                    --m;
                }
                tcol[m+1] = key;
                tval[m+1] = v;
            }
        });
        Gpu::streamSynchronize();
    } else {
        sort_rows(ncols, trow, tcol, tval, 1);
    }
}

// ---------------------------------------------------------------- SpMV

/**
 * y = A x. On GPU devices each row is handled by L work-items of one
 * sub-group, with L a power of two near the average row length (at most the
 * sub-group size); the partial sums are combined with sub-group shifts. On
 * the CPU device, and for complex values (AdaptiveCpp's group algorithms take
 * only scalar types), each row is handled by one work-item. Asynchronous.
 */
template <typename T, typename Index>
void spmv (Long nrows, T* AMREX_RESTRICT py, CsrView<T const,Index> const& A,
           T const* AMREX_RESTRICT px)
{
    auto const* mat = A.mat;
    auto const* col = A.col_index;
    auto const* row = A.row_offset;

    int L = 1;
    if constexpr (std::is_arithmetic_v<T>) {
        if (!on_cpu_device()) {
            auto const sg_sizes = Gpu::Device::syclDevice().get_info<sycl::info::device::sub_group_sizes>();
            int const sg_max = sg_sizes.empty() ? 1 : int(*std::max_element(sg_sizes.begin(), sg_sizes.end()));
            double const avg = double(A.nnz) / double(std::max(nrows, Long(1)));
            while (2*L <= std::min(avg, double(sg_max))) { L *= 2; }
            if (L <= 2) { L = 1; }   // the row loop is faster for short rows
        }
    }

    if (L == 1) {
        ParallelFor(nrows, [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            T r = 0;
            for (Long j = row[i]; j < row[i+1]; ++j) {
                r += mat[j] * px[col[j]];
            }
            py[i] = r;
        });
        return;
    }

    if constexpr (std::is_arithmetic_v<T>) {
        amrex::launch<wg_size>(num_blocks(nrows*L), 0, Gpu::gpuStream(),
        [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
        {
            auto const& sg = h.item->get_sub_group();
            Long const gid = Long(h.blockIdx())*wg_size + Long(h.threadIdx());
            Long const i = gid / L;
            int const lane = int(gid % L);
            T s = 0;
            if (i < nrows) {
                for (Long j = row[i] + lane; j < row[i+1]; j += L) {
                    s += mat[j] * px[col[j]];
                }
            }
            for (int off = L/2; off > 0; off /= 2) {
                s += sycl::shift_group_left(sg, s, off);
            }
            if (i < nrows && lane == 0) { py[i] = s; }
        });
    }
}

// ---------------------------------------------------------------- SpGEMM

/*
 * Hash-table SpGEMM, C = A B, for GPU devices.
 *
 * 1. Products per row of C (nprod), and a table class per row.
 * 2. Symbolic pass: the distinct columns of each row are counted in a hash
 *    table of keys of size >= 2 nprod.
 * 3. Row offsets of C by an exclusive scan.
 * 4. Numeric pass: values are accumulated in a table of keys and values of
 *    size >= 2 nnz(row), compacted, and written in column order.
 *
 * Rows are grouped into classes by table size (32 << class). A class's rows
 * are handled by teams of L work-items, rpw teams per work-group, each team
 * with its table in local memory. Rows whose table does not fit in local
 * memory form the last class: one work-group per row and a table in global
 * memory.
 */

//! Local-memory budget of the symbolic tables (keys only), in bytes.
constexpr std::size_t symbolic_budget = 32768;
//! Local-memory budget of the numeric tables (keys, values, compact copy), in bytes.
constexpr std::size_t numeric_budget = 36864;
//! Smallest table.
constexpr Long min_table = 32;

//! Smallest power of two >= n and >= min_table.
inline Long table_size (Long n)
{
    return std::max(min_table, Long(std::bit_ceil(std::uint64_t(std::max(n, Long(1))))));
}

//! Table entries that fit the budget, as a power of two.
template <typename T, typename Index>
constexpr Long numeric_entries ()
{
    // keys + values, plus a compact copy of half the size
    constexpr std::size_t bytes_per_entry = (sizeof(Index) + sizeof(T)) * 3 / 2;
    return Long(std::bit_floor(numeric_budget / bytes_per_entry));
}

template <typename Index>
constexpr Long symbolic_entries ()
{
    return Long(std::bit_floor(symbolic_budget / sizeof(Index)));
}

//! Classes 0..n-1 hold tables of min_table << class entries, up to `entries`;
//! one more class holds the rows that need a global-memory table.
inline int num_classes (Long entries)
{
    return int(std::countr_zero(std::uint64_t(entries / min_table))) + 2;
}

//! Class of a table of at least n entries; the last class if none fits.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int table_class (Long n, int nclass)
{
    int c = 0;
    while (c < nclass-1 && (min_table << c) < n) { ++c; }
    return c;
}

//! Key of an empty table slot. Column indices are never negative.
template <typename Index>
constexpr Index empty_key = Index(-1);

//! Hash of a column index into a table whose size is a power of two.
template <typename Index>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int hash_slot (Index c, int size)
{
    return int((std::uint32_t(c) * 2654435761u) & std::uint32_t(size-1));
}

//! Rows grouped by class.
struct RowBins
{
    std::vector<int> count;      //!< rows in each class
    std::vector<int> offset;     //!< first row of each class in `rows` (nclass+1 entries)
    Gpu::DeviceVector<int> rows; //!< row indices, class by class
};

/**
 * Groups rows by class. cls[i] < 0 means row i has nothing to do. Each
 * work-group counts its rows per class in local memory and reserves its
 * ranges with one global atomic per class. Synchronous.
 */
inline RowBins bin_rows (int const* cls, Long nrows, int nclass)
{
    AMREX_ALWAYS_ASSERT(nclass > 0 && nclass <= wg_size);
    RowBins bins;
    Gpu::DeviceVector<int> cnt(nclass, 0);
    auto* pcnt = cnt.data();
    int const nblocks = num_blocks(nrows);
    check_local_memory(2*std::size_t(nclass)*sizeof(int));

    amrex::launch<wg_size>(nblocks, std::size_t(nclass)*sizeof(int), Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
    {
        auto* lcnt = static_cast<int*>(h.local);
        int const lid = int(h.threadIdx());
        if (lid < nclass) { lcnt[lid] = 0; }
        sycl::group_barrier(h.item->get_group());
        Long const i = Long(h.blockIdx())*wg_size + lid;
        if (i < nrows && cls[i] >= 0) { atomic_inc<address_space::local_space>(lcnt + cls[i]); }
        sycl::group_barrier(h.item->get_group());
        if (lid < nclass && lcnt[lid] > 0) { Gpu::Atomic::AddNoRet(pcnt + lid, lcnt[lid]); }
    });
    bins.count.resize(nclass);
    Gpu::copyAsync(Gpu::deviceToHost, cnt.begin(), cnt.end(), bins.count.begin());
    Gpu::streamSynchronize();
    bins.offset.assign(nclass+1, 0);
    for (int c = 0; c < nclass; ++c) { bins.offset[c+1] = bins.offset[c] + bins.count[c]; }

    // The class offsets become the cursors of the scatter.
    Gpu::copyAsync(Gpu::hostToDevice, bins.offset.begin(), bins.offset.begin()+nclass, cnt.begin());
    bins.rows.resize(bins.offset[nclass]);
    auto* prows = bins.rows.data();
    amrex::launch<wg_size>(nblocks, 2*std::size_t(nclass)*sizeof(int), Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
    {
        auto* lcnt = static_cast<int*>(h.local);
        auto* lbase = lcnt + nclass;
        int const lid = int(h.threadIdx());
        if (lid < nclass) { lcnt[lid] = 0; }
        sycl::group_barrier(h.item->get_group());
        Long const i = Long(h.blockIdx())*wg_size + lid;
        int const c = (i < nrows) ? cls[i] : -1;
        int const slot = (c >= 0) ? atomic_inc<address_space::local_space>(lcnt + c) : 0;
        sycl::group_barrier(h.item->get_group());
        if (lid < nclass && lcnt[lid] > 0) { lbase[lid] = Gpu::Atomic::Add(pcnt + lid, lcnt[lid]); }
        sycl::group_barrier(h.item->get_group());
        if (c >= 0) { prows[lbase[c] + slot] = int(i); }
    });
    return bins;
}

/**
 * Offsets of the global-memory tables of the rows in the last class, one
 * table of size_of_row(row) entries each. Returns the total number of
 * entries. Synchronous.
 */
template <typename F>
Long global_table_offsets (RowBins const& bins, F const& size_of_row,
                           Gpu::DeviceVector<Long>& offsets)
{
    int const nclass = int(bins.count.size());
    int const n = bins.count[nclass-1];
    std::vector<int> rows(n);
    Gpu::copyAsync(Gpu::deviceToHost, bins.rows.begin() + bins.offset[nclass-1],
                   bins.rows.begin() + bins.offset[nclass], rows.begin());
    Gpu::streamSynchronize();
    std::vector<Long> off(n+1, 0);
    for (int k = 0; k < n; ++k) {
        Long const sz = size_of_row(rows[k]);
        AMREX_ALWAYS_ASSERT(sz < Long(std::numeric_limits<int>::max()));
        off[k+1] = off[k] + sz;
    }
    offsets.resize(n+1);
    Gpu::copyAsync(Gpu::hostToDevice, off.begin(), off.end(), offsets.begin());
    return off[n];
}

//! Work-items per row in a class: rpw teams of L lanes; lb lanes share a B row.
struct TeamLayout
{
    int rpw;
    int L;
    int lb;
};

/**
 * Team layout of a class whose tables have `table` entries out of a budget of
 * `entries`: as many teams as tables fit in the budget, at most one per
 * work-item.
 */
inline TeamLayout team_layout (Long entries, Long table, int lb_max)
{
    TeamLayout lay;
    lay.rpw = int(std::clamp(entries / table, Long(1), Long(wg_size)));
    lay.L = wg_size / lay.rpw;
    lay.lb = std::min(lay.L, lb_max);
    return lay;
}

//! The operands of C = A B and the product-loop shape chosen for them.
template <typename T, typename Index>
struct SpgemmOperands
{
    Index const* ar; Index const* ac; T const* av;
    Index const* br; Index const* bc; T const* bv;
    //! True: a lane takes every L-th product of the row (every lane busy);
    //! false: lb lanes stride each B row and L/lb A entries run at a time.
    bool flat;
};

/**
 * Calls f(p, q) for the products a(r,ac[p]) * b(ac[p],bc[q]) of row r that
 * fall to this lane. The flat loop keeps every lane busy on long B rows; the
 * nested loop avoids its per-product pointer chasing on short B rows.
 */
template <typename T, typename Index, typename F>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void for_each_product (SpgemmOperands<T,Index> const& o, Index r, int lane,
                       TeamLayout const& lay, F const& f)
{
    Index const pend = o.ar[r+1];
    if (o.flat) {
        Index p = o.ar[r];
        Index rem = lane;   // product index within the current B row
        Index qb = 0, bl = 0;
        auto load_brow = [&] () { Index const k = o.ac[p]; qb = o.br[k]; bl = o.br[k+1] - qb; };
        while (p < pend) {
            load_brow();
            if (rem < bl) { break; }
            rem -= bl; ++p;
        }
        while (p < pend) {
            f(p, qb + rem);
            rem += lay.L;
            while (rem >= bl) {
                rem -= bl; ++p;
                if (p >= pend) { break; }
                load_brow();
            }
        }
    } else {
        int const la = lane / lay.lb;
        int const lbb = lane % lay.lb;
        for (Index p = o.ar[r] + la; p < pend; p += lay.L/lay.lb) {
            Index const k = o.ac[p];
            for (Index q = o.br[k] + lbb; q < o.br[k+1]; q += lay.lb) { f(p, q); }
        }
    }
}

//! Global-memory tables of the last class: `offset` entries per row of the class.
template <typename T, typename Index>
struct GlobalTables
{
    Long const* offset = nullptr;
    Index* keys = nullptr;
    T* vals = nullptr;
    Index* ckeys = nullptr;
    T* cvals = nullptr;
    int* count = nullptr;
};

//! Hash table of keys for one row of the symbolic pass.
template <address_space AS, typename Index>
struct KeyTable
{
    Index* keys;
    int* count;
    int size;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void clear (int lane, int L) const noexcept
    {
        for (int s = lane; s < size; s += L) { keys[s] = empty_key<Index>; }
        if (lane == 0) { *count = 0; }
    }

    //! Inserts column c, counting it once.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void insert (Index c) const noexcept
    {
        int s = hash_slot(c, size);
        while (true) {
            Index const old = atomic_cas<AS>(keys + s, empty_key<Index>, c);
            if (old == empty_key<Index>) { atomic_inc<AS>(count); break; }
            if (old == c) { break; }
            s = (s + 1) & (size-1);
        }
    }
};

//! Hash table of keys and values for one row of the numeric pass, with a
//! compact copy of half its size for the output.
template <address_space AS, typename T, typename Index>
struct KeyValueTable
{
    Index* keys;
    T* vals;
    Index* ckeys;
    T* cvals;
    int* count;
    int size;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void clear (int lane, int L) const noexcept
    {
        for (int s = lane; s < size; s += L) { keys[s] = empty_key<Index>; vals[s] = T(0); }
        if (lane == 0) { *count = 0; }
    }

    //! Adds v to the value of column c.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void add (Index c, T v) const noexcept
    {
        int s = hash_slot(c, size);
        while (true) {
            Index const old = atomic_cas<AS>(keys + s, empty_key<Index>, c);
            if (old == empty_key<Index> || old == c) {
                atomic_add<AS>(vals + s, v);
                break;
            }
            s = (s + 1) & (size-1);
        }
    }

    //! Moves the occupied slots to the compact copy (needs a barrier after).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void compact (int lane, int L) const noexcept
    {
        for (int s = lane; s < size; s += L) {
            if (keys[s] != empty_key<Index>) {
                int const j = atomic_inc<AS>(count);
                ckeys[j] = keys[s];
                cvals[j] = vals[s];
            }
        }
    }

    //! Writes the compact copy in column order to ccol/cval + o.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void write_sorted (int lane, int L, Index* ccol, T* cval, Long o) const noexcept
    {
        int const n = *count;
        for (int j = lane; j < n; j += L) {
            Index const key = ckeys[j];
            int rank = 0;
            for (int m = 0; m < n; ++m) { rank += (ckeys[m] < key) ? 1 : 0; }
            ccol[o+rank] = key;
            cval[o+rank] = cvals[j];
        }
    }
};

//! Storage of the symbolic tables of one work-group, carved in this order.
template <typename Index, typename Arena>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto symbolic_storage (Arena& arena, int rpw, Long table)
{
    struct { Index* keys; int* count; } s;
    s.keys = arena.template take<Index>(std::size_t(rpw)*table);
    s.count = arena.template take<int>(rpw);
    return s;
}

//! Storage of the numeric tables of one work-group, carved in this order.
template <typename T, typename Index, typename Arena>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto numeric_storage (Arena& arena, int rpw, Long table)
{
    struct { T* vals; T* cvals; Index* keys; Index* ckeys; int* count; } s;
    s.vals = arena.template take<T>(std::size_t(rpw)*table);
    s.cvals = arena.template take<T>(std::size_t(rpw)*(table/2));
    s.keys = arena.template take<Index>(std::size_t(rpw)*table);
    s.ckeys = arena.template take<Index>(std::size_t(rpw)*(table/2));
    s.count = arena.template take<int>(rpw);
    return s;
}

//! Launch arguments of one class of one pass.
struct ClassLaunch
{
    int const* rows;   //!< rows of the class
    int nbin;          //!< number of rows
    Long table;        //!< table size; 0 for the global-memory class
    TeamLayout lay;
};

/**
 * Symbolic pass over one class: rnnz[r] = number of distinct columns of row
 * r. AS is the memory of the tables: local, or global for the last class.
 */
template <address_space AS, typename T, typename Index>
void symbolic_pass (SpgemmOperands<T,Index> const& o, ClassLaunch const& cl,
                    GlobalTables<T,Index> const& g, Index* rnnz)
{
    constexpr bool global = (AS == address_space::global_space);
    auto const lay = cl.lay;
    Long const table = cl.table;
    LocalBytes bytes;
    if (!global) { symbolic_storage<Index>(bytes, lay.rpw, table); }
    check_local_memory(bytes.n);
    int const ngroups = (cl.nbin + lay.rpw - 1) / lay.rpw;
    int const* rows = cl.rows;
    int const nbin = cl.nbin;

    amrex::launch<wg_size>(ngroups, bytes.n, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
    {
        int const lid = int(h.threadIdx());
        int const t = lid / lay.L;
        int const lane = lid % lay.L;
        Long const idx = Long(h.blockIdx())*lay.rpw + t;
        int const r = (idx < nbin) ? rows[idx] : -1;
        KeyTable<AS,Index> tab;
        if constexpr (global) {
            tab = {g.keys + g.offset[idx], g.count + idx, int(g.offset[idx+1] - g.offset[idx])};
        } else {
            LocalArena arena{static_cast<char*>(h.local)};
            auto const s = symbolic_storage<Index>(arena, lay.rpw, table);
            tab = {s.keys + t*table, s.count + t, int(table)};
        }
        if (r >= 0) { tab.clear(lane, lay.L); }
        sycl::group_barrier(h.item->get_group());
        if (r >= 0) {
            for_each_product(o, Index(r), lane, lay, [&] (Index, Index q) { tab.insert(o.bc[q]); });
        }
        sycl::group_barrier(h.item->get_group());
        if (r >= 0 && lane == 0) { rnnz[r] = Index(*tab.count); }
    });
}

/**
 * Numeric pass over one class: row r of C, whose offsets are crow, gets its
 * values accumulated in a table and written in column order.
 */
template <address_space AS, typename T, typename Index>
void numeric_pass (SpgemmOperands<T,Index> const& o, ClassLaunch const& cl,
                   GlobalTables<T,Index> const& g,
                   Index const* crow, Index* ccol, T* cval)
{
    constexpr bool global = (AS == address_space::global_space);
    auto const lay = cl.lay;
    Long const table = cl.table;
    LocalBytes bytes;
    if (!global) { numeric_storage<T,Index>(bytes, lay.rpw, table); }
    check_local_memory(bytes.n);
    int const ngroups = (cl.nbin + lay.rpw - 1) / lay.rpw;
    int const* rows = cl.rows;
    int const nbin = cl.nbin;

    amrex::launch<wg_size>(ngroups, bytes.n, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE (Gpu::Handler const& h) noexcept
    {
        int const lid = int(h.threadIdx());
        int const t = lid / lay.L;
        int const lane = lid % lay.L;
        Long const idx = Long(h.blockIdx())*lay.rpw + t;
        int const r = (idx < nbin) ? rows[idx] : -1;
        KeyValueTable<AS,T,Index> tab;
        if constexpr (global) {
            Long const b = g.offset[idx];
            tab = {g.keys + b, g.vals + b, g.ckeys + b, g.cvals + b, g.count + idx,
                   int(g.offset[idx+1] - b)};
        } else {
            LocalArena arena{static_cast<char*>(h.local)};
            auto const s = numeric_storage<T,Index>(arena, lay.rpw, table);
            tab = {s.keys + t*table, s.vals + t*table, s.ckeys + t*(table/2),
                   s.cvals + t*(table/2), s.count + t, int(table)};
        }
        if (r >= 0) { tab.clear(lane, lay.L); }
        sycl::group_barrier(h.item->get_group());
        if (r >= 0) {
            for_each_product(o, Index(r), lane, lay,
                             [&] (Index p, Index q) { tab.add(o.bc[q], o.av[p] * o.bv[q]); });
        }
        sycl::group_barrier(h.item->get_group());
        if (r >= 0) { tab.compact(lane, lay.L); }
        sycl::group_barrier(h.item->get_group());
        if (r >= 0) { tab.write_sorted(lane, lay.L, ccol, cval, Long(crow[r])); }
    });
}

/**
 * C = A B on a GPU device (see the description above). A and B must have
 * sorted or unsorted rows with non-negative column indices; C's rows are
 * sorted. Throws OutOfMemoryError from the arena if the temporaries do not
 * fit. Synchronous.
 *
 * \param nrows rows of A and C
 * \param ncols columns of B and C (unused by the kernels; kept for the caller's interface)
 */
template <typename T, template<typename> class V, typename Index>
CSR<T,V,Index> spgemm (Long nrows, Long ncols,
                       CsrView<T const,Index> const& A, CsrView<T const,Index> const& B)
{
    amrex::ignore_unused(ncols);
    AMREX_ALWAYS_ASSERT(nrows <= Long(std::numeric_limits<int>::max()));

    constexpr Long sym_entries = symbolic_entries<Index>();
    constexpr Long num_entries = numeric_entries<T,Index>();
    int const nclass_sym = num_classes(sym_entries);
    int const nclass_num = num_classes(num_entries);

    // Product-loop shape from the average B row length: see for_each_product.
    double const avg_b = double(B.nnz) / double(std::max(B.nrows, Long(1)));
    SpgemmOperands<T,Index> const o{A.row_offset, A.col_index, A.mat,
                                    B.row_offset, B.col_index, B.mat,
                                    avg_b >= 8.0};
    int lb_max = 1;
    while (lb_max < 8 && double(lb_max) < avg_b) { lb_max *= 2; }

    // Runs one pass over all classes; the last class uses global memory.
    auto run_classes = [&] (RowBins const& bins, int nclass, Long entries, auto const& pass)
    {
        for (int k = 0; k < nclass; ++k) {
            if (bins.count[k] == 0) { continue; }
            bool const global = (k == nclass-1);
            Long const table = global ? 0 : (min_table << k);
            ClassLaunch const cl{bins.rows.data() + bins.offset[k], bins.count[k], table,
                                 team_layout(entries, global ? entries : table, lb_max)};
            // The global-memory kernels index their tables by work-group.
            AMREX_ALWAYS_ASSERT(!global || cl.lay.rpw == 1);
            if (global) {
                pass(std::integral_constant<address_space, address_space::global_space>{}, cl);
            } else {
                pass(std::integral_constant<address_space, address_space::local_space>{}, cl);
            }
        }
        Gpu::streamSynchronize();
    };

    // 1. Products per row and symbolic class.
    Gpu::DeviceVector<Long> nprod(nrows);
    Gpu::DeviceVector<int> cls(nrows);
    auto* pnprod = nprod.data();
    auto* pcls = cls.data();
    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (Long i) noexcept
    {
        Long n = 0;
        for (Index p = o.ar[i]; p < o.ar[i+1]; ++p) { n += o.br[o.ac[p]+1] - o.br[o.ac[p]]; }
        pnprod[i] = n;
        pcls[i] = (n > 0) ? table_class(2*n, nclass_sym) : -1;
    });

    // 2. Symbolic pass.
    Gpu::DeviceVector<Index> rnnz(nrows+1, 0);
    auto* prnnz = rnnz.data();
    {
        auto const bins = bin_rows(pcls, nrows, nclass_sym);
        Gpu::DeviceVector<Long> goff;
        Gpu::DeviceVector<Index> gkeys;
        Gpu::DeviceVector<int> gcnt;
        if (bins.count[nclass_sym-1] > 0) {
            std::vector<Long> nprod_h(nrows);
            Gpu::copyAsync(Gpu::deviceToHost, nprod.begin(), nprod.end(), nprod_h.begin());
            Gpu::streamSynchronize();
            Long const total = global_table_offsets(bins,
                [&] (int r) { return table_size(2*nprod_h[r]); }, goff);
            gkeys.resize(total);
            gcnt.resize(bins.count[nclass_sym-1]);
        }
        GlobalTables<T,Index> g;
        g.offset = goff.data(); g.keys = gkeys.data(); g.count = gcnt.data();
        run_classes(bins, nclass_sym, sym_entries, [&] (auto as, ClassLaunch const& cl) {
            symbolic_pass<decltype(as)::value>(o, cl, g, prnnz);
        });
    }

    // 3. Row offsets.
    CSR<T,V,Index> C;
    C.row_offset.resize(nrows+1);
    Long const nnz = Scan::ExclusiveSum(nrows+1, prnnz, C.row_offset.data(), Scan::retSum);
    AMREX_ALWAYS_ASSERT(nnz <= Long(std::numeric_limits<Index>::max()));
    C.col_index.resize(nnz);
    C.mat.resize(nnz);
    C.nnz = nnz;
    if (nnz == 0) { return C; }

    // 4. Numeric pass, classes by the row length of C.
    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (Long i) noexcept
    {
        Long const n = prnnz[i];
        pcls[i] = (n > 0) ? table_class(2*n, nclass_num) : -1;
    });
    {
        auto const bins = bin_rows(pcls, nrows, nclass_num);
        Gpu::DeviceVector<Long> goff;
        Gpu::DeviceVector<Index> gkeys, gckeys;
        Gpu::DeviceVector<T> gvals, gcvals;
        Gpu::DeviceVector<int> gcnt;
        if (bins.count[nclass_num-1] > 0) {
            std::vector<Index> rnnz_h(nrows);
            Gpu::copyAsync(Gpu::deviceToHost, rnnz.begin(), rnnz.begin()+nrows, rnnz_h.begin());
            Gpu::streamSynchronize();
            Long const total = global_table_offsets(bins,
                [&] (int r) { return table_size(2*Long(rnnz_h[r])); }, goff);
            gkeys.resize(total);
            gvals.resize(total);
            gckeys.resize(total);
            gcvals.resize(total);
            gcnt.resize(bins.count[nclass_num-1]);
        }
        GlobalTables<T,Index> const g{goff.data(), gkeys.data(), gvals.data(),
                                      gckeys.data(), gcvals.data(), gcnt.data()};
        Index const* crow = C.row_offset.data();
        Index* ccol = C.col_index.data();
        T* cval = C.mat.data();
        run_classes(bins, nclass_num, num_entries, [&] (auto as, ClassLaunch const& cl) {
            numeric_pass<decltype(as)::value>(o, cl, g, crow, ccol, cval);
        });
    }

    return C;
}

} // namespace detail::sp_acpp
} // namespace amrex

#endif
#endif
