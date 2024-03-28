#pragma once
#ifndef MATHLIB_POLYNOMIAL_HPP
#define MATHLIB_POLYNOMIAL_HPP

#include <cstddef>
#include <vector>
#include <cmath>
#include <concepts>
#include "lalib/type_traits.hpp"
#include "lalib/ops/vec_ops.hpp"

namespace mathlib {

template<typename T>
struct Polynomial {
    Polynomial(std::vector<T>&& coeffs) noexcept;

    auto n_orders() const noexcept -> size_t;
    auto operator()(double s) const noexcept -> T;
    auto deriv(double s) const noexcept -> T;

private:
    std::vector<T> _coeffs;
};


// ## Internal functor templates ## //
namespace _polynomial_ {

template<typename T> struct _PolynomialPoint_ { static_assert([](){ return false; }(), "Unsupported polynomial type."); };
template<typename T> struct _PolynomialDeriv_ { static_assert([](){ return false; }(), "Unsupported polynomial type."); };

template<std::floating_point T>
struct _PolynomialPoint_<T> {
    static auto point(double s, const std::vector<T>& coeffs) noexcept -> T {
        auto p = coeffs.front();
        for (auto i = 1u; i < coeffs.size(); ++i) {
            p += std::pow(s, i) * coeffs[i];
        }
        return p;
    }
};

template<lalib::Vector T>
struct _PolynomialPoint_<T> {
    static auto point(double s, const std::vector<T>& coeffs) noexcept -> T {
        auto p = coeffs.front();
        for (auto i = 1u; i < coeffs.size(); ++i) {
            lalib::axpy(std::pow(s, i), coeffs[i], p);
        }
        return p;
    }
};

template<std::floating_point T>
struct _PolynomialDeriv_<T> {
    static auto deriv(double s, const std::vector<T>& coeffs) noexcept -> T {
        auto deriv = 0.0;
        for (auto i = 1u; i < coeffs.size(); ++i) {
            deriv += i * std::pow(s, i - 1) * coeffs[i];
        }
        return deriv;
    }
};

template<std::floating_point T, size_t N>
struct _PolynomialDeriv_<lalib::SizedVec<T, N>> {
    static auto deriv(double s, const std::vector<lalib::SizedVec<T, N>>& coeffs) noexcept -> lalib::SizedVec<T, N> {
        auto deriv = lalib::SizedVec<T, N>::filled(0.0);
        for (auto i = 1u; i < coeffs.size(); ++i) {
            lalib::axpy(i * std::pow(s, i - 1), coeffs[i], deriv);
        }
        return deriv;
    }
};

template<std::floating_point T>
struct _PolynomialDeriv_<lalib::DynVec<T>> {
    static auto deriv(double s, const std::vector<lalib::DynVec<T>>& coeffs) noexcept -> lalib::DynVec<T> {
        auto n = coeffs.front().size();
        auto deriv = lalib::DynVec<T>::filled(n, 0.0);
        for (auto i = 1u; i < coeffs.size(); ++i) {
            lalib::axpy(i * std::pow(s, i - 1), coeffs[i], deriv);
        }
        return deriv;
    }
};

}


// ====== Implementation ====== // 

template<typename T>
Polynomial<T>::Polynomial(std::vector<T>&& coeffs) noexcept: 
    _coeffs(std::move(coeffs))
{ }


template<typename T>
auto Polynomial<T>::n_orders() const noexcept -> size_t {
    return this->_coeffs.size() - 1;
}

template<typename T>
inline auto Polynomial<T>::operator()(double s) const noexcept -> T {
    auto p = _polynomial_::_PolynomialPoint_<T>::point(s, this->_coeffs);
    return p;
}

template<typename T>
inline auto Polynomial<T>::deriv(double s) const noexcept -> T {
    auto deriv = _polynomial_::_PolynomialDeriv_<T>::deriv(s, this->_coeffs);
    return deriv;
}

}

#endif