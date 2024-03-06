#include <gtest/gtest.h>
#include <functional>
#include <cmath>
#include "mathlib/roots/secant.hpp"

TEST(SecantTests, RootFindingTest) {
    auto solver = mathlib::Secant(1e-6);

    auto x1 = solver.find_root([](double x){ return std::pow(x, 6) + x - 2; }, 10, 9);
    ASSERT_TRUE(x1);
    ASSERT_NEAR(1.0, x1.value(), 1e-6);

    auto x2 = solver.find_root_info((double(*)(double))std::atan, 3, -3);
    ASSERT_TRUE(x2);
    ASSERT_NEAR(0.0, x2.final_value(), 1e-6);
}