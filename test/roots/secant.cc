#include <gtest/gtest.h>
#include <functional>
#include "../../include/roots/secant.hpp"

TEST(SecantTests, RootFindingTest) {
    auto solver = mathlib::Secant(1e-6);
    auto x = solver.find_root([](double x){ return std::pow(x, 6) + x - 2; }, 10, 9);

    ASSERT_NEAR(1.0, x.value(), 1e-6);
}