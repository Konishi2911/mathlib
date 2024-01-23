#pragma once
#ifndef MATHLIB_NLP_NELDER_MEAD_HPP
#define MATHLIB_NLP_NELDER_MEAD_HPP

#include "cost_function.hpp"
#include "solver_common.hpp"
#include <concepts>
#include <algorithm>
#include <numeric>
#include <ranges>

namespace mathlib::nlp {

struct NelderMead {
    NelderMead(double tol) noexcept;

    template<typename T, std::invocable<T> F>
    auto solve(T x1, T x2, F&& func, size_t max_iter) const -> NlpResult<T>;

private:
    double _tol;
    double _alpha = 1.0;
    double _gamma = 2.0;
    double _rho = 0.5;
    double _sigma = 0.5;
};

inline NelderMead::NelderMead(double tol) noexcept:
    _tol(tol)
{ }

template<typename T, std::invocable<T> F>
inline auto NelderMead::solve(T x1, T x2, F &&func, size_t max_iter) const -> NlpResult<T>
{
    constexpr size_t dim = 1u;
    auto verts = std::vector<std::pair<double, T>>{
        std::make_pair(func(x1), x1),
        std::make_pair(func(x2), x2)
    };

    double err = 0.0;
    size_t k = 0u; 
    for (; k <= max_iter; ++k) {
        // Concergence chack
        err = std::abs(verts.front().second - verts.back().second);
        if (err < this->_tol) {
            return NlpResult(true, k, verts.front().second, verts.front().first, err);
        }

        // Order
        std::ranges::sort(verts, {}, &std::pair<double, T>::first);
        auto nodes = std::views::all(verts) | std::views::transform([](auto a) { return a.second; });
        auto x_o = std::accumulate(nodes.begin(), nodes.end() - 1, 0.0);
        x_o /= dim; 

        // Reflection
        auto dx = x_o - verts.back().second;
        auto x_r = x_o + this->_alpha * (dx);
        auto f_r = func(x_r);

        // Expansion
        if (f_r < verts.front().first) {
            auto dx = (x_r - x_o);
            auto x_e = x_o + this->_gamma * dx;
            auto f_e = func(x_e);
            
            if (f_e < f_r) {
                verts.back() = std::make_pair(f_e, x_e);
            } else {
                verts.back() = std::make_pair(f_r, x_r);
            }
            continue;
        }
        // Continue
        else if (f_r < (*(verts.end() - 2)).first) { 
            continue; 
        }
        // Contraction
        else {
            if (f_r < verts.back().first) {
                auto dx = (x_r - x_o);
                auto x_c = x_o + this->_rho * dx;
                auto f_c = func(x_c);
                if (f_c < f_r) {
                    verts.back() = std::make_pair(f_c, x_c);
                    continue;
                }
            }
        }
        // Shrink
        for (auto i = 1u; i < dim + 1; ++i) {
            auto dx = verts[i].second;
            auto x = verts.front().second + this->_sigma * dx;
            verts[i] = std::make_pair(func(x), x);
        }
    }
    return NlpResult(false, k, verts.front().second, verts.front().first, err);
}

}

#endif