#pragma once
#ifndef MATHLIB_NLP_SOLVER_CONCEPT_HPP
#define MATHLIB_NLP_SOLVER_CONCEPT_HPP

#include <concepts>
#include <cstdint>
#include <numeric>
#include "lalib/vec.hpp"

namespace mathlib::nlp {

template<typename T>
struct NlpResult {
    NlpResult(bool is_converged, uint64_t iter, T sol, double final_cost, double final_residual) noexcept;

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

namespace _internal_ {
    template<std::floating_point T>
    auto infinity(const T&) noexcept -> T {
        return std::numeric_limits<T>::infinity();
    }
    
    template<std::floating_point T>
    auto infinity(const lalib::DynVec<T>& x) noexcept -> lalib::DynVec<T> {
        return lalib::DynVec<T>::filled(x.size(), std::numeric_limits<T>::infinity());
    }

    template<std::floating_point T>
    auto params_resi2(const T& prev, const T& curr) noexcept -> double {
        return std::pow(curr - prev, 2);
    }

    template<std::floating_point T>
    auto params_resi2(const lalib::DynVec<T> &prev, const lalib::DynVec<T> &curr) noexcept -> double {
        auto resi = curr - prev;
        auto resi_tot = resi.dot(resi) / resi.size();
        return resi_tot;
    }

    template<std::floating_point T>
    auto grad_crit(const T& grad) noexcept -> double {
        return grad;
    }

    template<std::floating_point T>
    auto grad_crit(const lalib::DynVec<T> &grad) noexcept -> double {
        auto gc = grad.norm2() / grad.size();
        return gc;
    }
}

// ==== Info ==== //

template<typename T>
inline NlpResult<T>::NlpResult(bool is_converged, uint64_t iter, T sol, double final_cost, double final_err) noexcept:
    _is_converged(is_converged), _iter(iter), _sol(sol), _final_cost(final_cost), _err(final_err)
{ }

template<typename T>
inline auto NlpResult<T>::is_converged() const noexcept -> bool
{
    return this->_is_converged;
}

template<typename T>
inline auto NlpResult<T>::final_iter() const noexcept -> uint64_t
{
    return this->_iter;
}

template <typename T>
inline auto NlpResult<T>::sol() const noexcept -> T
{
    return this->_sol;
}

template<typename T>
inline auto NlpResult<T>::final_cost() const noexcept -> double
{
    return this->_final_cost;
}

template<typename T>
inline auto NlpResult<T>::final_error() const noexcept -> double
{
    return this->_err;
}

template<typename T>
inline NlpResult<T>::operator bool() const noexcept
{
    return this->_is_converged;
}


}

#endif