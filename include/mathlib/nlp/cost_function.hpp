#pragma once
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


template<std::floating_point T, std::invocable<T> F>
requires std::same_as<decltype(T(std::declval<T>())), double>
struct NumericCostFunc {
    using Grad = double;

    NumericCostFunc(F&& func, T&& dx) noexcept;

    auto operator()(const T& x) const -> double;
    auto grad(const T& x) const -> Grad;

private:
    F _func;
    T _dx;
};

template <std::floating_point T, std::invocable<T> F>
requires std::same_as<decltype(T(std::declval<T>())), double>
inline NumericCostFunc<T, F>::NumericCostFunc(F &&func, T&& dx) noexcept:
    _func(std::move(func)), _dx(dx)
{
}

template <std::floating_point T, std::invocable<T> F>
requires std::same_as<decltype(T(std::declval<T>())), double>
inline auto NumericCostFunc<T, F>::operator()(const T &x) const -> double
{
    return _func(x);
}
template <std::floating_point T, std::invocable<T> F>
requires std::same_as<decltype(T(std::declval<T>())), double>
inline auto NumericCostFunc<T, F>::grad(const T &x) const -> Grad
{
    auto df = this->_func(x + this->_dx) - this->_func(x - this->_dx);
    df /= (2.0 * this->_dx);
    return df;
}

}

#endif