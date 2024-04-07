#include <gtest/gtest.h>
#include "mathlib/nlp/nelder_mead.hpp"


auto quadrature_func(double x) noexcept -> double {
    return (x - 2.0)*(x + 1.0);
}

TEST(NelderMeadTests, QuadratureFuncTests) {
    auto solver = mathlib::nlp::NelderMead(1e-6);
    auto cost_func = mathlib::nlp::NumericCostFunc<double, double(*)(double)>(quadrature_func, 1e-5, 1e-4);

    static_assert(mathlib::nlp::CostFunc<decltype(cost_func), double>);

    {
        auto sol = solver.solve(0.0, 0.1, std::move(cost_func), 1000);
        ASSERT_TRUE(sol);
        EXPECT_NEAR(sol.sol(), 0.5, 1e-6);
    }

    {
        auto sol = solver.solve(9.0, 10.0, std::move(cost_func), 1000);
        ASSERT_TRUE(sol);
        EXPECT_NEAR(sol.sol(), 0.5, 1e-6);
    }
}