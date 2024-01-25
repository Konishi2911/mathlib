#include <gtest/gtest.h>
#include "../../include/interpolation/linear.hpp"
#include <iostream>

TEST(LinearInterpolationTest, InterpolationTest) {
    auto data = std::vector {
        std::make_pair(0.0, 1.0),
        std::make_pair(0.5, -1.0),
        std::make_pair(1.0, 4.0),
        std::make_pair(2.0, 3.0),
        std::make_pair(4.0, 7.0),
    };
    auto f = mathlib::intrpl::Linear(data);

    ASSERT_EQ(4, f.n_segments());

    EXPECT_DOUBLE_EQ(1.0, f(0.0));
    EXPECT_DOUBLE_EQ(5.0, f(3.0));
    EXPECT_DOUBLE_EQ(7.0, f(4.0));
}

TEST(LinearInterpolationTest, DuplicatedPointInterpolationTest) {
    auto data = std::vector {
        std::make_pair(0.0, 1.0),
        std::make_pair(0.5, -1.0),
        std::make_pair(0.5, -2.0),
        std::make_pair(1.0, 4.0),
        std::make_pair(2.0, 3.0),
        std::make_pair(4.0, 7.0),
    };
    auto f = mathlib::intrpl::Linear(data);

    ASSERT_EQ(5, f.n_segments());

    EXPECT_DOUBLE_EQ(1.0, f(0.0));
    EXPECT_DOUBLE_EQ(-2.0, f(0.5)); // If duplicated point is specified, the last value among the duplicated point will be returned.
    EXPECT_DOUBLE_EQ(5.0, f(3.0));
    EXPECT_DOUBLE_EQ(7.0, f(4.0));
}