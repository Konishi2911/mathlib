#include <gtest/gtest.h>
#include "mathlib/interpolation/cubic_spline.hpp"
#include <iostream>

template<typename T>
void print_spline(const mathlib::CubicSpline<T>& spline, size_t n) noexcept {
	for (auto i = 0u; i <= n; ++i) {
        auto x = (spline.domain().first * (n - i) + spline.domain().second * (i)) / static_cast<double>(n);
		std::cout << x << ", " << spline(x) << std::endl;
	}
}

TEST(CubicSplineTests, InterpolationTest) {
    auto data = std::vector {
        std::make_pair(0.0, 1.0),
        std::make_pair(0.5, -1.0),
        std::make_pair(1.0, 4.0),
        std::make_pair(2.0, 3.0),
        std::make_pair(4.0, 7.0),
    };
    auto f = mathlib::CubicSpline(data);

    ASSERT_EQ(4, f.n_segments());

    EXPECT_DOUBLE_EQ(1.0, f(0.0));
    EXPECT_DOUBLE_EQ(7.0, f(4.0));

    print_spline(f, 100);
}

TEST(CubicSplineTests, InterpolationDualTest) {
    auto data = std::vector {
        std::make_pair(0.0, 1.0),
        std::make_pair(0.5, -1.0),
        std::make_pair(1.0, 4.0),
        std::make_pair(2.0, 3.0),
        std::make_pair(4.0, 7.0),
    };
    auto f = mathlib::CubicSpline(data);

    ASSERT_EQ(4, f.n_segments());

    auto x0 = mathlib::Dual(0.0, 1.0);
    auto x1 = mathlib::Dual(4.0, 1.0);

    EXPECT_DOUBLE_EQ(1.0, f(x0).a);
    EXPECT_DOUBLE_EQ(7.0, f(x1).a);
}

TEST(CubicSplineTests, InterpolationHyperDualTest) {
    auto data = std::vector {
        std::make_pair(0.0, 1.0),
        std::make_pair(0.5, -1.0),
        std::make_pair(1.0, 4.0),
        std::make_pair(2.0, 3.0),
        std::make_pair(4.0, 7.0),
    };
    auto f = mathlib::CubicSpline(data);

    ASSERT_EQ(4, f.n_segments());

    auto x0 = mathlib::HyperDual2(0.0, 1.0, 1.0, 0.0);
    auto x1 = mathlib::HyperDual2(4.0, 1.0, 1.0, 0.0);

    EXPECT_DOUBLE_EQ(1.0, f(x0).a);
    EXPECT_DOUBLE_EQ(0.0, f(x0).d);
    
    EXPECT_DOUBLE_EQ(7.0, f(x1).a);
    EXPECT_DOUBLE_EQ(0.0, f(x1).d);
}