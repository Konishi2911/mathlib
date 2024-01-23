#pragma once
#ifndef MATHLIB_NLP_SOLVER_CONCEPT_HPP
#define MATHLIB_NLP_SOLVER_CONCEPT_HPP

#include <concepts>

namespace mathlib::nlp {

template<typename T>
struct NlpResult {
    NlpResult(bool is_converged, uint64_t iter, T sol, double final_cost, double final_err) noexcept;

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