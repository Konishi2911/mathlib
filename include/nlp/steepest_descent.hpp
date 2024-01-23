#pragma once
#ifndef MATHLIB_NLP_STEEPEST_DESCENT_HPP
#define MATHLIB_NLP_STEEPEST_DESCENT_HPP

#include <optional>
#include "cost_function.hpp"
#include "../../third_party/lalib/include/vec.hpp"

namespace mathlib::nlp {

struct SteepestDescent {
    template<typename T>
    struct Info {
        Info(bool is_converged, uint64_t iter, T sol, double final_cost, double final_err) noexcept;

        auto is_converged() const noexcept -> bool;
        auto final_iter() const noexcept -> uint64_t;
        auto sol() const noexcept -> T;
        auto final_cost() const noexcept -> double;
        auto final_error() const noexcept -> double;

        explicit operator bool() const noexcept;

    private:
        bool _is_converged;
        uint64_t _iter;
        T _sol;
        double _final_cost;
        double _err;
    };

    SteepestDescent(double tol) noexcept;

    template<typename T, CostFunc<T> F>
    auto solve(T init, F&& func, size_t max_iter) const -> Info<T>;

private:
    double _tol;
};


SteepestDescent::SteepestDescent(double tol) noexcept:
    _tol(tol)
{ }

template <typename T, CostFunc<T> F>
inline auto SteepestDescent::solve(T x, F &&func, size_t max_iter) const -> Info<T>
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
            auto info = Info(true, i, new_x, cost, err);
            return info;
        } else {
            x = new_x;
            cost = new_cost;
        }
    }
    return Info(false, i, x, cost, err);
}


// ==== SteepestDescent::Info ==== //

template<typename T>
inline SteepestDescent::Info<T>::Info(bool is_converged, uint64_t iter, T sol, double final_cost, double final_err) noexcept:
    _is_converged(is_converged), _iter(iter), _sol(sol), _final_cost(final_cost), _err(final_err)
{ }

template<typename T>
inline auto SteepestDescent::Info<T>::is_converged() const noexcept -> bool
{
    return this->_is_converged;
}

template<typename T>
inline auto SteepestDescent::Info<T>::final_iter() const noexcept -> uint64_t
{
    return this->_iter;
}

template <typename T>
inline auto SteepestDescent::Info<T>::sol() const noexcept -> T
{
    return this->_sol;
}

template<typename T>
inline auto SteepestDescent::Info<T>::final_cost() const noexcept -> double
{
    return this->_final_cost;
}

template<typename T>
inline auto SteepestDescent::Info<T>::final_error() const noexcept -> double
{
    return this->_err;
}

template<typename T>
inline SteepestDescent::Info<T>::operator bool() const noexcept
{
    return this->_is_converged;
}

}

#endif