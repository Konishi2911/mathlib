#pragma once
#ifndef MATHLIB_ROOTS_SECANT_HPP
#define MATHLIB_ROOTS_SECANT_HPP

#include <concepts>
#include <optional>

namespace mathlib {

struct Secant {
    struct Info {
        Info(bool is_converged, uint64_t iter, double final_val, double final_err) noexcept;

        auto is_converged() const noexcept -> bool;
        auto final_iter() const noexcept -> uint64_t;
        auto final_value() const noexcept -> double;
        auto final_error() const noexcept -> double;

        explicit operator bool() const noexcept;

    private:
        bool _is_converged;
        uint64_t _iter;
        double _final;
        double _err;

    };

    Secant(double error) noexcept;

    template<std::invocable<double> F>
    auto find_root(F&& f, double x1, double x2, size_t max_iter = 100) const noexcept(noexcept(f(std::declval<double>()))) -> std::optional<double>;

    template<std::invocable<double> F>
    auto find_root_info(F&& f, double x1, double x2, size_t max_iter = 100) const noexcept(noexcept(f(std::declval<double>()))) -> Info;

private:
    double _err;

    template<std::invocable<double> F>
    auto __find_root_step_impl(F&& f, double x1, double x2) const noexcept -> double;
};

inline Secant::Secant(double error) noexcept: 
    _err(error)
{ }

template<std::invocable<double> F>
inline auto Secant::find_root(F&& f, double x1, double x2, size_t max_iter) const noexcept(noexcept(f(std::declval<double>()))) -> std::optional<double> {
    for (auto k = 0u; k < max_iter; ++k) {
        auto x_next = this->__find_root_step_impl(f, x1, x2);

        auto err = std::abs(f(x_next));
        if (err < this->_err) {
            return x_next;
        } else {
            x1 = x2;
            x2 = x_next;
        }
    }
    return std::nullopt;
}

template <std::invocable<double> F>
inline auto Secant::find_root_info(F &&f, double x1, double x2, size_t max_iter) const noexcept(noexcept(f(std::declval<double>()))) -> Info
{
    auto x_next = 0.0;
    for (auto k = 0u; k < max_iter; ++k) {
        x_next = this->__find_root_step_impl(f, x1, x2);

        auto err = std::abs(f(x_next));
        if (err < this->_err) {
            auto info = Info(true, k, x_next, err);
            return info;
        } else {
            x1 = x2;
            x2 = x_next;
        }
    }
    return Info(false, max_iter, x_next, std::abs(f(x_next)));
}

template <std::invocable<double> F>
inline auto Secant::__find_root_step_impl(F &&f, double x1, double x2) const noexcept -> double
{
    auto x_next = x2 - f(x2) * (x2 - x1) / (f(x2) - f(x1));
    return x_next;
}


// ==== Secant::Info ==== //

inline Secant::Info::Info(bool is_converged, uint64_t iter, double final_val, double final_err) noexcept:
    _is_converged(is_converged), _iter(iter), _final(final_val), _err(final_err)
{ }

inline auto Secant::Info::is_converged() const noexcept -> bool
{
    return this->_is_converged;
}

inline auto Secant::Info::final_iter() const noexcept -> uint64_t
{
    return this->_iter;
}

inline auto Secant::Info::final_value() const noexcept -> double
{
    return this->_final;
}

inline auto Secant::Info::final_error() const noexcept -> double
{
    return this->_err;
}

inline Secant::Info::operator bool() const noexcept
{
    return this->_is_converged;
}

}

#endif