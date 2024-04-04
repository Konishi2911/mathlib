#pragma once
#include "lalib/mat/dyn_mat.hpp"
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

    NumericCostFunc(F&& func, T dx) noexcept: 
        _func(std::forward<F>(func)), 
        _dx(dx)
    {}

    auto operator()(const T& x) const -> double { 
        return this->_func(x); 
    }

    auto grad(const T& x) const -> Grad {
        auto df = (this->_func(x + this->_dx) - this->_func(x - this->_dx)) / (2.0 * this->_dx);
        return df;
    }

    auto hessian(const T& x) const -> T {
        auto ddf = (this->_func(x + this->_dx) - 2.0 * this->_func(x) + this->_func(x - this->_dx)) / (this->_dx * this->_dx);
        return ddf;
    }

private:
    F _func;
    T _dx;
};


template<std::floating_point T, std::invocable<lalib::DynVec<T>> F>
struct NumericCostFunc<lalib::DynVec<T>, F> {
    using Grad = lalib::DynVec<T>;

    NumericCostFunc(F&& func, T dx) noexcept: 
        _func(std::forward<F>(func)), 
        _dx(dx)
    {}

    auto operator()(const lalib::DynVec<T>& x) const -> double { 
        return this->_func(x); 
    }

    auto grad(const lalib::DynVec<T>& x) const -> Grad {
        auto grad = std::vector<double>();
        grad.reserve(x.size());
        for (auto i = 0u; i < x.size(); ++i) {
            auto xp = x;    xp[i] += this->_dx;
            auto xn = x;    xn[i] -= this->_dx;
            auto df = (this->_func(xp) - this->_func(xn)) / (2.0 * this->_dx);
            grad.emplace_back(std::move(df));
        }
        return lalib::DynVec<T>(std::move(grad));
    }

    auto hessian(const lalib::DynVec<T>& x) const -> lalib::DynMat<T> {
        auto n = x.size();
        auto hess = lalib::DynMat<T>::filled(0.0, n, n);
        for (auto i = 0u; i < x.size(); ++i) {
            for (auto j = 0u; j <= i; ++j) {
                auto x1 = x;    x1[i] += this->_dx;     x1[j] += this->_dx;
                auto x2 = x;    x2[i] += this->_dx;     x2[j] -= this->_dx;
                auto x3 = x;    x3[i] -= this->_dx;     x3[j] += this->_dx;
                auto x4 = x;    x4[i] -= this->_dx;     x4[j] -= this->_dx;

                hess(i, j) = ((this->_func(x1) + this->_func(x4)) - (this->_func(x2) + this->_func(x3))) / (4.0 * this->_dx * this->_dx);
                hess(j, i) = hess(i, j);
            }
        }
        return hess;
    }

private:
    F _func;
    T _dx;
};

}

#endif