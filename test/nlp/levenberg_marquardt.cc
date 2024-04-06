#include <gtest/gtest.h>
#include "mathlib/nlp/levenberg_marquardt.hpp"

auto quadrature_func(double x) noexcept -> double {
    return (x - 2.0)*(x + 1.0);
}

auto rosenbrock_func(const lalib::DynVecD& x) noexcept -> double {
    auto n = x.size();
    auto r = 0.0;
    for (auto i = 0u; i < n - 1; ++i) {
        r += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1.0 - x[i], 2);
    }
    return r;
}

TEST(LevenbergMarquardtTests, QuadratureFuncTests) {
    auto solver = mathlib::nlp::LevenbregMarquardt();
    auto cost_func = mathlib::nlp::NumericCostFunc<double, double(*)(double)>(quadrature_func, 1e-5);

    static_assert(mathlib::nlp::CostFunc<decltype(cost_func), double>);

    auto sol = solver.solve(0.0, std::move(cost_func), 100, 1e-6, 1e-6);
    ASSERT_TRUE(sol);
    EXPECT_NEAR(sol.sol(), 0.5, 2e-5);
    EXPECT_LE(sol.final_error(), 1e-6);
}

TEST(LevenbergMarquardtTests, RosenbrockFuncTests) {
    auto solver = mathlib::nlp::LevenbregMarquardt();
    auto cost_func = mathlib::nlp::NumericCostFunc<lalib::DynVecD, double(*)(const lalib::DynVecD&)>(rosenbrock_func, 1e-5);

    static_assert(mathlib::nlp::CostFunc<decltype(cost_func), lalib::DynVecD>);

    auto init = lalib::DynVecD::filled(3, 4.0);
    auto sol = solver.solve(init, std::move(cost_func), 100, 1e-6, 1e-6);
    ASSERT_TRUE(sol);
    EXPECT_NEAR(sol.sol()[0], 1.0, 2e-5);
    EXPECT_NEAR(sol.sol()[1], 1.0, 2e-5);
    EXPECT_NEAR(sol.sol()[2], 1.0, 2e-5);
    EXPECT_LE(sol.final_error(), 1e-6);
}