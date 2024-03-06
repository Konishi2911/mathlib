#pragma once 
#ifndef MATHLIB_INTEGRAL_SIMPSON_HPP
#define MATHLIB_INTEGRAL_SIMPSON_HPP

#include <functional>

namespace mathlib::integral {

struct Simpson {
    template <std::invocable<double> F>
    auto integrate(double st, double ed, F &&f, size_t n) const noexcept -> decltype(f(st));
};

template <std::invocable<double> F>
inline auto Simpson::integrate(double st, double ed, F &&f, size_t n) const noexcept -> decltype(f(st))
{
    auto ds = ed - st;
    auto itg = f(st) + f(ed);
    for (auto k = 1u; k <= 2 * n - 1; k += 2) {
        auto x = ds * static_cast<double>(k) / (2 * n) + st;
        itg += 4.0 * f(x);
    }
    for (auto k = 2u; k <= 2 * n - 2; k += 2) {
        auto x = ds * static_cast<double>(k) / (2 * n) + st;
        itg += 2.0 * f(x);
    }
    itg = itg * ds / (6.0 * n);
    return itg;
}

}

#endif