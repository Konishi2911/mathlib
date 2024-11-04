#include <gtest/gtest.h>
#include "mathlib/ode/runge_kutta.hpp"
#include <cmath>
#include <iostream>

auto f1(double y, double t) noexcept -> double {
    return y - t*t + 1;
}

// Reference: https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
auto f1_exact(double t) noexcept -> double {
    return t*t + 2 * t + 1 - 0.5 * std::exp(t);
}

TEST(RKTests, SolTest) {
    auto rk = mathlib::ode::RungeKutta(0.0, 0.5, std::function<double(double, double)>(f1));

    for (auto i = 0u; i < 10; ++i) {
        rk.advance(0.1);
        ASSERT_NEAR(f1_exact(rk.time()), rk.x(), 1e-4);
    }
}