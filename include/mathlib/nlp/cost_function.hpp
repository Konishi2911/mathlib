#pragma once
#include "lalib/mat/dyn_mat.hpp"
#include <limits>
#ifndef MATHLIB_NLP_COST_FUNCTION_HPP
#define MATHLIB_NLP_COST_FUNCTION_HPP

#include <concepts>
#include "lalib/vec.hpp"

namespace mathlib::nlp {

template<typename T, typename P>
concept CostFunc = requires(const T& t, const P& param) {
    typename T::Grad;
    { t(param) } -> std::convertible_to<double>;
    { t.grad(param) } -> std::convertible_to<typename T::Grad>;
};

template<typename T, std::invocable<T> F> struct NumericCostFunc {
    NumericCostFunc() = delete;
};

template<std::floating_point T, std::invocable<T> F>
struct NumericCostFunc<T, F> {
    using Grad = T;

    NumericCostFunc(F&& func, T grad_dx, T hess_dx) noexcept: 
        _func(std::forward<F>(func)), 
        _grad_dx(grad_dx),
        _hess_dx(hess_dx)
    {}

    auto operator()(const T& x) const -> double { 
        return this->_func(x); 
    }

    auto grad(const T& x) const -> Grad {
        auto df = (this->_func(x + this->_grad_dx) - this->_func(x - this->_grad_dx)) / (2.0 * this->_grad_dx);
        return df;
    }

    auto hessian(const T& x) const -> T {
        auto ddf = (this->_func(x + this->_hess_dx) - 2.0 * this->_func(x) + this->_func(x - this->_hess_dx)) / (this->_hess_dx * this->_hess_dx);
        return ddf;
    }

private:
    F _func;
    T _grad_dx;
    T _hess_dx;
};


template<std::floating_point T, std::invocable<lalib::DynVec<T>> F>
struct NumericCostFunc<lalib::DynVec<T>, F> {
    using Grad = lalib::DynVec<T>;

    NumericCostFunc(F&& func, T grad_dx, T hess_dx) noexcept: 
        _func(std::forward<F>(func)), 
        _grad_dx(grad_dx),
        _hess_dx(hess_dx)
    {}

    auto operator()(const lalib::DynVec<T>& x) const -> double { 
        return this->_func(x); 
    }

    auto grad(const lalib::DynVec<T>& x) const -> Grad {
        auto grad = std::vector<double>(x.size(), 0.0);
        #pragma omp parallel for
        for (auto i = 0u; i < x.size(); ++i) {
            auto rdx = this->_grad_dx;
            auto xp = x;    xp[i] += rdx;
            auto xn = x;    xn[i] -= rdx;
            auto fp = this->_func(xp);
            auto fn = this->_func(xn);
            auto df = 0.5 * (fp / rdx - fn / rdx);
            grad[i] = std::move(df);
        }
        return lalib::DynVec<T>(std::move(grad));
    }

    auto hessian(const lalib::DynVec<T>& x) const -> lalib::DynMat<T> {
        auto n = x.size();
        auto hess = lalib::DynMat<T>::filled(0.0, n, n);
        #pragma omp parallel for
        for (auto i = 0u; i < x.size(); ++i) {
            auto rdx = this->_hess_dx;
            for (auto j = 0u; j <= i; ++j) {
                auto x1 = x;    x1[i] += rdx;     x1[j] += rdx;
                auto x2 = x;    x2[i] += rdx;     x2[j] -= rdx;
                auto x3 = x;    x3[i] -= rdx;     x3[j] += rdx;
                auto x4 = x;    x4[i] -= rdx;     x4[j] -= rdx;
                auto f1 = this->_func(x1);
                auto f2 = this->_func(x2);
                auto f3 = this->_func(x3);
                auto f4 = this->_func(x4);

                auto ddx = rdx * rdx;
                hess(i, j) = 0.25 * ((f1 + f4) / ddx - (f2 + f3) / ddx);
                hess(j, i) = hess(i, j);
            }
        }
        return hess;
    }

private:
    F _func;
    T _grad_dx;
    T _hess_dx;
};

}

#endif