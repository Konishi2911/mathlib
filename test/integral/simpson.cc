#include <gtest/gtest.h>
#include <numbers>
#include <cmath>
#include "../../include/integral/simpson.hpp"

TEST(SimpsonIntegralTests, IntegralTest) {
    auto int1 = mathlib::integral::Simpson().integrate(
        0.0, std::numbers::pi / 2.0,
        [](auto x) {
            return std::sin(x) / (1.0 + std::cos(x));
        }, 
        10
    );
    EXPECT_NEAR(std::log(2.0), int1, 1e-6);
}