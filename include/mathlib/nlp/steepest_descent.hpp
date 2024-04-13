#pragma once
#include <algorithm>
#include <limits>
#ifndef MATHLIB_NLP_STEEPEST_DESCENT_HPP
#define MATHLIB_NLP_STEEPEST_DESCENT_HPP

#include <optional>
#include "mathlib/nlp/cost_function.hpp"
#include "mathlib/nlp/solver_common.hpp"
#include "lalib/vec.hpp"

namespace mathlib::nlp {

struct SteepestDescent {
    SteepestDescent() noexcept;

    template<typename T, CostFunc<T> F>
    auto solve(T init, F&& func, size_t max_iter, double cost_resi, double params_resi, double grad_crit) const -> NlpResult<T>;

private:
};

namespace _sd_ {
    template<std::floating_point T>
    inline auto armijo(double xi, double alpha, T gd, T d) noexcept -> T {
        auto a = xi * alpha * gd * d;
        return a;
    }

    template<std::floating_point T>
    inline auto armijo(double xi, double alpha, const lalib::DynVec<T>& gd, const lalib::DynVec<T>& d) noexcept -> T {
        auto a = xi * alpha * gd.dot(d);
        return a;
    }

    template<std::floating_point T>
    inline auto direction(const T& grad) noexcept -> T {
        auto d = -grad / std::abs(grad);
        return d;
    }

    template<std::floating_point T>
    inline auto direction(const lalib::DynVec<T>& grad) noexcept -> lalib::DynVec<T> {
        auto d = -grad / grad.norm2();
        return d;
    }
}


inline SteepestDescent::SteepestDescent() noexcept
{ }

template <typename T, CostFunc<T> F>
inline auto SteepestDescent::solve(T x, F &&func, size_t max_iter, double cost_resi, double params_resi, double grad_crit) const -> NlpResult<T>
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
        auto d = _sd_::direction(gd);

        auto new_x = x + alpha * d;
        auto new_cost = func(new_x);
        while (new_cost - cost > _sd_::armijo(xi, alpha, gd, d)) {
            alpha *= tau;
            new_x = x + alpha * d;
            new_cost = func(new_x);
        }

        // Convergence check
        auto resi_x = _internal_::params_resi2(x, new_x);
        auto resi = std::abs(new_cost - cost);
        auto gc = _internal_::grad_crit(gd);
        if (resi < cost_resi && resi_x < params_resi && gc < grad_crit) { 
            auto result = NlpResult<T>(true, i, x, cost, resi); 
            return result;
        } else {
            // Update process
            x = new_x;
            cost = new_cost;
        }
    }
    return NlpResult(false, i, x, cost, err);
}


}

#endif