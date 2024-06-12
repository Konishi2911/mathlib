#pragma once
#ifndef MATHLIB_ODE_RUNGE_KUTTA
#define MATHLIB_ODE_RUNGE_KUTTA

#include <concepts>
#include <memory>

namespace mathlib::ode {

template<std::floating_point T, class U, std::invocable<U, T> F>
struct RungeKutta {
    /// @brief  Construct RK solver for the given function
    RungeKutta(T&& init_time, U&& init, F&& f) noexcept;

    /// @brief  Returns the current simulation time.
    auto time() const noexcept -> T;

    /// @brief  Returns the current parameter.
    auto x() const noexcept -> U;

    /// @brief  Advances single time step with given time step width.
    void advance(T dt) noexcept;

private:
    T _time;
    U _x;
    F _f;
};


// === Implementaion === //
template<std::floating_point T, class U, std::invocable<U, T> F>
RungeKutta<T, U, F>::RungeKutta(T&& init_time, U&& init, F&& f) noexcept: 
    _time(std::move(init_time)), _x(std::move(init)), _f(std::move(f))
{}

template<std::floating_point T, class U, std::invocable<U, T> F>
auto RungeKutta<T, U, F>::time() const noexcept -> T {
    return this->_time;
}

template<std::floating_point T, class U, std::invocable<U, T> F>
auto RungeKutta<T, U, F>::x() const noexcept -> U {
    return this->_x;
}

template<std::floating_point T, class U, std::invocable<U, T> F>
void RungeKutta<T, U, F>::advance(T dt) noexcept {
    auto&& k1 = this->_f(this->_x, this->_time);
    auto&& k2 = this->_f(this->_x + 0.5 * dt * k1, this->_time + 0.5 * dt);
    auto&& k3 = this->_f(this->_x + 0.5 * dt * k2, this->_time + 0.5 * dt);
    auto&& k4 = this->_f(this->_x + dt * k3, this->_time + dt);

    auto dx = (k1 + 2 * (k2 + k3) + k4) * dt / 6.0;
    this->_x = this->_x + dx;
    this->_time = this->_time + dt;
}

}
#endif