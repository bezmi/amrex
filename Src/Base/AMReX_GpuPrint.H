#ifndef AMREX_GPU_PRINT_H_
#define AMREX_GPU_PRINT_H_
#include <AMReX_Config.H>

#include <cstdio>

#ifdef AMREX_USE_SYCL
#  include <sycl/sycl.hpp>
#endif

#if defined(AMREX_USE_SYCL) && defined(__ACPP__)

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <type_traits>

namespace amrex::detail {

// AdaptiveCpp's generic target can print only a plain string from device
// code (__acpp_sscp_print). AMREX_DEVICE_PRINTF therefore formats the message
// itself in device code. Each argument is formatted according to its C++
// type; length modifiers (h, l, ll, z, j, t) in the format are skipped.
// Supported conversions: d i u x X o c s p f F e E g G %, with the flags
// '-', '+', ' ', '0', a field width and a precision. Output longer than the
// buffer is truncated.

struct AcppPrintBuffer
{
    char* m_buf;
    std::size_t m_cap;
    std::size_t m_n = 0;

    void put (char c) noexcept {
        if (m_n + 1 < m_cap) { m_buf[m_n++] = c; }
    }
    void finish () noexcept { m_buf[(m_n < m_cap) ? m_n : m_cap-1] = '\0'; }
};

struct AcppPrintSpec
{
    bool left = false;
    bool plus = false;
    bool space = false;
    bool zero = false;
    int width = 0;
    int precision = -1;
    char conv = 'd';
};

// Writes the characters [s, s+len) padded to the field width.
inline void acpp_print_padded (AcppPrintBuffer& out, AcppPrintSpec const& spec,
                               char sign, char const* s, int len) noexcept
{
    int const total = len + (sign ? 1 : 0);
    int const pad = (spec.width > total) ? spec.width - total : 0;
    if (!spec.left && !spec.zero) { for (int i = 0; i < pad; ++i) { out.put(' '); } }
    if (sign) { out.put(sign); }
    if (!spec.left && spec.zero) { for (int i = 0; i < pad; ++i) { out.put('0'); } }
    for (int i = 0; i < len; ++i) { out.put(s[i]); }
    if (spec.left) { for (int i = 0; i < pad; ++i) { out.put(' '); } }
}

inline void acpp_print_unsigned (AcppPrintBuffer& out, AcppPrintSpec const& spec,
                                 char sign, unsigned long long v) noexcept
{
    unsigned base = 10;
    if (spec.conv == 'x' || spec.conv == 'X' || spec.conv == 'p') { base = 16; }
    if (spec.conv == 'o') { base = 8; }
    char const* digits = (spec.conv == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
    char tmp[24];
    int n = 0;
    do { tmp[n++] = digits[v % base]; v /= base; } while (v != 0);
    char s[24];
    for (int i = 0; i < n; ++i) { s[i] = tmp[n-1-i]; }
    acpp_print_padded(out, spec, sign, s, n);
}

inline void acpp_print_signed (AcppPrintBuffer& out, AcppPrintSpec const& spec,
                               long long v) noexcept
{
    char sign = spec.plus ? '+' : (spec.space ? ' ' : '\0');
    unsigned long long u = static_cast<unsigned long long>(v);
    if (v < 0) { sign = '-'; u = 0ULL - u; }
    acpp_print_unsigned(out, spec, sign, u);
}

// Fixed (%f) or scientific (%e) digits of a finite, non-negative x with
// `prec` digits after the decimal point. Returns the number of characters.
inline int acpp_format_double (char* s, double x, int prec, bool sci, char e_char) noexcept
{
    int exp10 = 0;
    if (sci && x != 0.0) {
        while (x >= 10.0) { x /= 10.0; ++exp10; }
        while (x < 1.0) { x *= 10.0; --exp10; }
    }
    double r = 0.5;
    for (int i = 0; i < prec; ++i) { r /= 10.0; }
    x += r;
    if (sci && x >= 10.0) { x /= 10.0; ++exp10; }
    int n = 0;
    if (x >= 1.8e19) {
        // Too large for a 64-bit integer part: print 17 significant digits
        // (all that a double holds) followed by zeros.
        int e = 0;
        while (x >= 10.0) { x /= 10.0; ++e; }
        for (int i = 0; i <= e; ++i) {
            int d = 0;
            if (i < 17) {
                d = static_cast<int>(x);
                if (d > 9) { d = 9; }
                x = (x - d) * 10.0;
            }
            s[n++] = static_cast<char>('0' + d);
        }
        x = 0.0;
    } else {
        auto ip = static_cast<unsigned long long>(x);
        x -= static_cast<double>(ip);
        char tmp[24];
        int m = 0;
        do { tmp[m++] = static_cast<char>('0' + ip % 10); ip /= 10; } while (ip != 0);
        while (m > 0) { s[n++] = tmp[--m]; }
    }
    if (prec > 0) { s[n++] = '.'; }
    for (int i = 0; i < prec; ++i) {
        x *= 10.0;
        auto d = static_cast<int>(x);
        if (d > 9) { d = 9; }
        s[n++] = static_cast<char>('0' + d);
        x -= d;
    }
    if (sci) {
        s[n++] = e_char;
        s[n++] = (exp10 < 0) ? '-' : '+';
        int ae = (exp10 < 0) ? -exp10 : exp10;
        if (ae >= 100) { s[n++] = static_cast<char>('0' + ae / 100); }
        s[n++] = static_cast<char>('0' + (ae / 10) % 10);
        s[n++] = static_cast<char>('0' + ae % 10);
    }
    return n;
}

inline void acpp_print_double (AcppPrintBuffer& out, AcppPrintSpec const& spec, double x) noexcept
{
    char sign = spec.plus ? '+' : (spec.space ? ' ' : '\0');
    if (x < 0.0 || (x == 0.0 && 1.0/x < 0.0)) { sign = '-'; x = -x; }
    bool const upper = (spec.conv == 'F' || spec.conv == 'E' || spec.conv == 'G');
    AcppPrintSpec sp = spec;
    if (x != x) {
        sp.zero = false;
        acpp_print_padded(out, sp, '\0', upper ? "NAN" : "nan", 3);
        return;
    }
    if (x > 1.7976931348623157e308) {
        sp.zero = false;
        acpp_print_padded(out, sp, sign, upper ? "INF" : "inf", 3);
        return;
    }
    int prec = (spec.precision < 0) ? 6 : spec.precision;
    char s[400];
    int n = 0;
    char const lc = static_cast<char>(spec.conv | 0x20);
    if (lc == 'f') {
        n = acpp_format_double(s, x, prec, false, 'e');
    } else if (lc == 'e') {
        n = acpp_format_double(s, x, prec, true, upper ? 'E' : 'e');
    } else { // g
        if (prec == 0) { prec = 1; }
        // Exponent X that %e would print with precision prec-1.
        char t[40];
        int const nt = acpp_format_double(t, x, prec-1, true, 'e');
        int X = 0;
        int i = nt - 1;
        int scale = 1;
        while (t[i] != '+' && t[i] != '-') { X += (t[i] - '0') * scale; scale *= 10; --i; }
        if (t[i] == '-') { X = -X; }
        if (prec > X && X >= -4) {
            n = acpp_format_double(s, x, prec-1-X, false, 'e');
        } else {
            n = acpp_format_double(s, x, prec-1, true, upper ? 'E' : 'e');
        }
        // Remove trailing zeros of the fraction (and the point), as %g does.
        int epos = n;
        for (int k = 0; k < n; ++k) { if (s[k] == 'e' || s[k] == 'E') { epos = k; break; } }
        bool has_point = false;
        for (int k = 0; k < epos; ++k) { if (s[k] == '.') { has_point = true; } }
        if (has_point) {
            int end = epos;
            while (s[end-1] == '0') { --end; }
            if (s[end-1] == '.') { --end; }
            int const removed = epos - end;
            for (int k = epos; k < n; ++k) { s[k-removed] = s[k]; }
            n -= removed;
        }
    }
    acpp_print_padded(out, sp, sign, s, n);
}

template <typename T>
void acpp_print_arg (AcppPrintBuffer& out, AcppPrintSpec const& spec, T const& v) noexcept
{
    using U = std::decay_t<T>;
    if constexpr (std::is_same_v<U, char>) {
        if (spec.conv == 'c') {
            acpp_print_padded(out, spec, '\0', &v, 1);
        } else {
            acpp_print_signed(out, spec, v);
        }
    } else if constexpr (std::is_same_v<U, bool>) {
        acpp_print_unsigned(out, spec, '\0', v ? 1ULL : 0ULL);
    } else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) {
        if (spec.conv == 'c') {
            char const c = static_cast<char>(v);
            acpp_print_padded(out, spec, '\0', &c, 1);
        } else if constexpr (std::is_signed_v<U>) {
            if (spec.conv == 'd' || spec.conv == 'i') {
                acpp_print_signed(out, spec, static_cast<long long>(v));
            } else {
                acpp_print_unsigned(out, spec, '\0', static_cast<unsigned long long>(v));
            }
        } else {
            acpp_print_unsigned(out, spec, '\0', static_cast<unsigned long long>(v));
        }
    } else if constexpr (std::is_floating_point_v<U>) {
        acpp_print_double(out, spec, static_cast<double>(v));
    } else if constexpr (std::is_same_v<U, char const*> || std::is_same_v<U, char*>) {
        char const* str = (v != nullptr) ? v : "(null)";
        int len = 0;
        while (str[len] != '\0' && (spec.precision < 0 || len < spec.precision)) { ++len; }
        AcppPrintSpec sp = spec;
        sp.zero = false;
        acpp_print_padded(out, sp, '\0', str, len);
    } else if constexpr (std::is_pointer_v<U>) {
        AcppPrintSpec sp = spec;
        sp.conv = 'p';
        out.put('0'); out.put('x');
        acpp_print_unsigned(out, sp, '\0', reinterpret_cast<unsigned long long>(v));
    } else {
        static_assert(std::is_arithmetic_v<U>, "AMREX_DEVICE_PRINTF: unsupported argument type");
    }
}

// Copies fmt up to the next conversion, which it parses into spec. Returns
// the position after the conversion, or nullptr at the end of fmt.
inline char const* acpp_print_next (AcppPrintBuffer& out, char const* fmt,
                                    AcppPrintSpec& spec) noexcept
{
    while (*fmt != '\0') {
        if (*fmt != '%') { out.put(*fmt++); continue; }
        ++fmt;
        if (*fmt == '%') { out.put('%'); ++fmt; continue; }
        spec = AcppPrintSpec{};
        for (bool flag = true; flag; ) {
            switch (*fmt) {
            case '-': spec.left = true; ++fmt; break;
            case '+': spec.plus = true; ++fmt; break;
            case ' ': spec.space = true; ++fmt; break;
            case '0': spec.zero = true; ++fmt; break;
            case '#': ++fmt; break;
            default: flag = false;
            }
        }
        while (*fmt >= '0' && *fmt <= '9') { spec.width = spec.width*10 + (*fmt++ - '0'); }
        if (*fmt == '.') {
            ++fmt;
            spec.precision = 0;
            while (*fmt >= '0' && *fmt <= '9') { spec.precision = spec.precision*10 + (*fmt++ - '0'); }
        }
        while (*fmt == 'h' || *fmt == 'l' || *fmt == 'L' || *fmt == 'z' || *fmt == 'j' || *fmt == 't') { ++fmt; }
        if (*fmt == '\0') { return nullptr; }
        spec.conv = *fmt++;
        if (spec.left) { spec.zero = false; }
        return fmt;
    }
    return nullptr;
}

inline void acpp_format (AcppPrintBuffer& out, char const* fmt) noexcept
{
    AcppPrintSpec spec;
    while (fmt != nullptr) {
        fmt = acpp_print_next(out, fmt, spec);  // no arguments left: print the rest as text
    }
}

template <typename T, typename... Ts>
void acpp_format (AcppPrintBuffer& out, char const* fmt, T const& first, Ts const&... rest) noexcept
{
    AcppPrintSpec spec;
    fmt = acpp_print_next(out, fmt, spec);
    if (fmt == nullptr) { return; }
    acpp_print_arg(out, spec, first);
    acpp_format(out, fmt, rest...);
}

template <typename... Args>
void acpp_device_printf (char const* fmt, Args const&... args) noexcept
{
    AMREX_IF_ON_HOST((
        if constexpr (sizeof...(Args) == 0) {
            std::fputs(fmt, stdout);
        } else {
            std::printf(fmt, args...);
        }
    ))
    AMREX_IF_ON_DEVICE((
        char buf[512];
        AcppPrintBuffer out{buf, sizeof(buf)};
        acpp_format(out, fmt, args...);
        out.finish();
        __acpp_sscp_print(buf);
    ))
}

}

#  define AMREX_DEVICE_PRINTF(...) amrex::detail::acpp_device_printf(__VA_ARGS__);
#elif defined(AMREX_USE_SYCL)
#  define AMREX_DEVICE_PRINTF(...) \
      sycl::ext::oneapi::experimental::printf(__VA_ARGS__);
#elif defined(AMREX_USE_CUDA)
#  define AMREX_DEVICE_PRINTF(...) std::printf(__VA_ARGS__);
#elif defined(AMREX_USE_HIP)
#  define AMREX_DEVICE_PRINTF(...) ::printf(__VA_ARGS__);
#else
#  define AMREX_DEVICE_PRINTF(...) std::printf(__VA_ARGS__);
#endif

#endif  // AMREX_GPU_PRINT_H_
