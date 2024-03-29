#include <gtest/gtest.h>
#include "mathlib/nlp/steepest_descent.hpp"


auto quadrature_func(double x) noexcept -> double {
    return (x - 2.0)*(x + 1.0);
}

TEST(SteepestDescentTest, QuadratureFuncTests) {
    auto solver = mathlib::nlp::SteepestDescent(1e-6);
    auto cost_func = mathlib::nlp::NumericCostFunc<double, double(*)(double)>(quadrature_func, 1e-5);

    static_assert(mathlib::nlp::CostFunc<decltype(cost_func), double>);

    auto sol = solver.solve(0.0, std::move(cost_func), 100);
    ASSERT_TRUE(sol);
    EXPECT_NEAR(sol.sol(), 0.5, 1e-6);
}