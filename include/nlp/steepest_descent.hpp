#pragma once
#ifndef MATHLIB_NLP_STEEPEST_DESCENT_HPP
#define MATHLIB_NLP_STEEPEST_DESCENT_HPP

#include <optional>
#include "cost_function.hpp"
#include "solver_common.hpp"
#include "../../third_party/lalib/include/vec.hpp"

namespace mathlib::nlp {

struct SteepestDescent {
    SteepestDescent(double tol) noexcept;

    template<typename T, CostFunc<T> F>
    auto solve(T init, F&& func, size_t max_iter) const -> NlpResult<T>;

private:
    double _tol;
};


SteepestDescent::SteepestDescent(double tol) noexcept:
    _tol(tol)
{ }

template <typename T, CostFunc<T> F>
inline auto SteepestDescent::solve(T x, F &&func, size_t max_iter) const -> NlpResult<T>
{
    auto i = 0u;
    auto cost = func(x);
    auto err = 0.0;
    for (; i < max_iter; ++i) {
        auto gd = func.grad(x);

        // Armijo process
        constexpr auto xi = 1.0e-4;
        constexpr auto tau = 0.2;
        auto alpha = 1.0;
        auto d = -std::copysign(1.0, gd);

        auto new_x = x + alpha * d;
        auto new_cost = func(new_x);
        while (new_cost - cost > xi * alpha * std::abs(gd) * d) {
            alpha *= tau;
            new_x = x + alpha * d;
            new_cost = func(new_x);
        }

        // Update process
        err = std::abs(new_x - x);
        if (err < this->_tol) {
            auto info = NlpResult(true, i, new_x, cost, err);
            return info;
        } else {
            x = new_x;
            cost = new_cost;
        }
    }
    return NlpResult(false, i, x, cost, err);
}


}

#endif