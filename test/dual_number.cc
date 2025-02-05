#include "mathlib/dual_number.hpp"
#include <gtest/gtest.h>

template<typename T>
auto f1(T x) -> T { return x * sin(x); }

template<typename T>
auto df1(T x) -> T { return sin(x) + x * cos(x); }

template<typename T>
auto ddf1(T x) -> T { return 2.0 * cos(x) - x * sin(x); }


TEST(DualNumberTests, DerivationTest1) {
    mathlib::Dual<double> x(1.0, 1.0);
    mathlib::Dual<double> y = f1(x);
    EXPECT_DOUBLE_EQ(y.a, f1(1.0));
    EXPECT_DOUBLE_EQ(y.b, df1(1.0));
}

TEST(DualNumberTests, SecDerivationTest1) {
    mathlib::HyperDual2<double> x(1.0, 1.0, 1.0, 0.0);
    mathlib::HyperDual2<double> y = f1(x);
    EXPECT_DOUBLE_EQ(y.a, f1(1.0));
    EXPECT_DOUBLE_EQ(y.b, df1(1.0));
    EXPECT_DOUBLE_EQ(y.c, df1(1.0));
    EXPECT_DOUBLE_EQ(y.d, ddf1(1.0));
}


TEST(DualNumberTests, DualNumberAddition) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    mathlib::Dual<double> z = x + y;
    EXPECT_DOUBLE_EQ(z.a, 4.0);
    EXPECT_DOUBLE_EQ(z.b, 6.0);
}

TEST(DualNumberTests, DualNumberSubtraction) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    mathlib::Dual<double> z = x - y;
    EXPECT_DOUBLE_EQ(z.a, -2.0);
    EXPECT_DOUBLE_EQ(z.b, -2.0);
}

TEST(DualNumberTests, DualNumberMultiplication) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    mathlib::Dual<double> z = x * y;
    EXPECT_DOUBLE_EQ(z.a, 3.0);
    EXPECT_DOUBLE_EQ(z.b, 10.0);
}

TEST(DualNumberTests, DualNumberDivision) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    mathlib::Dual<double> z = x / y;
    EXPECT_DOUBLE_EQ(z.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(z.b, (2.0 * 3.0 - 1.0 * 4.0) / (3.0 * 3.0));
}

TEST(DualNumberTests, DualNumberScalarMultiplication) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = 2.0 * x;
    EXPECT_DOUBLE_EQ(y.a, 2.0);
    EXPECT_DOUBLE_EQ(y.b, 4.0);
}

TEST(DualNumberTests, DualNumberScalarDivision) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = x / 2.0;
    EXPECT_DOUBLE_EQ(y.a, 0.5);
    EXPECT_DOUBLE_EQ(y.b, 1.0);
}

TEST(DualNumberTests, DualNumberUnaryMinus) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = -x;
    EXPECT_DOUBLE_EQ(y.a, -1.0);
    EXPECT_DOUBLE_EQ(y.b, -2.0);
}

TEST(DualNumberTests, DualNumberTan) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = tan(x);
    EXPECT_DOUBLE_EQ(y.a, std::tan(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (std::cos(1.0) * std::cos(1.0)));
}

TEST(DualNumberTests, DualNumberSinh) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = sinh(x);
    EXPECT_DOUBLE_EQ(y.a, std::sinh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 * std::cosh(1.0));
}

TEST(DualNumberTests, DualNumberCosh) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = cosh(x);
    EXPECT_DOUBLE_EQ(y.a, std::cosh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 * std::sinh(1.0));
}

TEST(DualNumberTests, DualNumberTanh) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = tanh(x);
    EXPECT_DOUBLE_EQ(y.a, std::tanh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (std::cosh(1.0) * std::cosh(1.0)));
}

TEST(DualNumberTests, DualNumberPowScalar) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = pow(x, 2.0);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0 * 2.0 / 1.0);
}

TEST(DualNumberTests, DualNumberPowDual) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> alpha(3.0, 4.0);
    mathlib::Dual<double> y = pow(x, alpha);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 1.0 * (4.0 * std::log(1.0) + 3.0 * 2.0 / 1.0));
}

TEST(DualNumberTests, DualNumberSqrt) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = sqrt(x);
    EXPECT_DOUBLE_EQ(y.a, std::sqrt(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (2.0 * std::sqrt(1.0)));
}

TEST(DualNumberTests, DualNumberComparison) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(1.0, 2.0);
    mathlib::Dual<double> z(2.0, 3.0);
    EXPECT_TRUE(x == y);
    EXPECT_FALSE(x == z);
    EXPECT_FALSE(x != y);
    EXPECT_TRUE(x != z);
}


TEST(DualNumberTests, DualNumberCopy) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(x);
    EXPECT_DOUBLE_EQ(x.a, y.a);
    EXPECT_DOUBLE_EQ(x.b, y.b);
}

TEST(DualNumberTests, DualNumberMove) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(std::move(x));
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}

TEST(DualNumberTests, DualNumberAssignment) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y;
    y = x;
    EXPECT_DOUBLE_EQ(x.a, y.a);
    EXPECT_DOUBLE_EQ(x.b, y.b);
}

TEST(DualNumberTests, DualNumberAssignmentMove) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y;
    y = std::move(x);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}


TEST(DualNumberTests, DualNumberAssignmentAddition) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    x += y;
    EXPECT_DOUBLE_EQ(x.a, 4.0);
    EXPECT_DOUBLE_EQ(x.b, 6.0);
}

TEST(DualNumberTests, DualNumberAssignmentSubtraction) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    x -= y;
    EXPECT_DOUBLE_EQ(x.a, -2.0);
    EXPECT_DOUBLE_EQ(x.b, -2.0);
}

TEST(DualNumberTests, DualNumberAssignmentMultiplication) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    x *= y;
    EXPECT_DOUBLE_EQ(x.a, 3.0);
    EXPECT_DOUBLE_EQ(x.b, 10.0);
}

TEST(DualNumberTests, DualNumberAssignmentDivision) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    x /= y;
    EXPECT_DOUBLE_EQ(x.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.b, (2.0 * 3.0 - 1.0 * 4.0) / (3.0 * 3.0));
}

TEST(DualNumberTests, DualNumberAssignmentScalarAddition) {
    mathlib::Dual<double> x(1.0, 2.0);
    x += 3.0;
    EXPECT_DOUBLE_EQ(x.a, 4.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0);
}

TEST(DualNumberTests, DualNumberAssignmentScalarSubtraction) {
    mathlib::Dual<double> x(1.0, 2.0);
    x -= 3.0;
    EXPECT_DOUBLE_EQ(x.a, -2.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0);
}

TEST(DualNumberTests, DualNumberAssignmentScalarMultiplication) {
    mathlib::Dual<double> x(1.0, 2.0);
    x *= 3.0;
    EXPECT_DOUBLE_EQ(x.a, 3.0);
    EXPECT_DOUBLE_EQ(x.b, 6.0);
}

TEST(DualNumberTests, DualNumberAssignmentScalarDivision) {
    mathlib::Dual<double> x(1.0, 2.0);
    x /= 3.0;
    EXPECT_DOUBLE_EQ(x.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0 / 3.0);
}

TEST(DualNumberTests, DualNumberAdditionScalar) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = x + 3.0;
    EXPECT_DOUBLE_EQ(y.a, 4.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}

TEST(DualNumberTests, DualNumberSubtractionScalar) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = x - 3.0;
    EXPECT_DOUBLE_EQ(y.a, -2.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}

TEST(DualNumberTests, DualNumberMultiplicationScalar) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = x * 3.0;
    EXPECT_DOUBLE_EQ(y.a, 3.0);
    EXPECT_DOUBLE_EQ(y.b, 6.0);
}

TEST(DualNumberTests, DualNumberDivisionScalar) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = x / 3.0;
    EXPECT_DOUBLE_EQ(y.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0 / 3.0);
}

TEST(DualNumberTests, DualNumberAdditionScalarReverse) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = 3.0 + x;
    EXPECT_DOUBLE_EQ(y.a, 4.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}

TEST(DualNumberTests, DualNumberSubtractionScalarReverse) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = 3.0 - x;
    EXPECT_DOUBLE_EQ(y.a, 2.0);
    EXPECT_DOUBLE_EQ(y.b, -2.0);
}

TEST(DualNumberTests, DualNumberMultiplicationScalarReverse) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = 3.0 * x;
    EXPECT_DOUBLE_EQ(y.a, 3.0);
    EXPECT_DOUBLE_EQ(y.b, 6.0);
}

TEST(DualNumberTests, DualNumberDivisionScalarReverse) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = 3.0 / x;
    EXPECT_DOUBLE_EQ(y.a, 3.0);
    EXPECT_DOUBLE_EQ(y.b, -6.0);
}

TEST(DualNumberTests, DualNumberScale) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::scale(3.0, x);
    EXPECT_DOUBLE_EQ(x.a, 3.0);
    EXPECT_DOUBLE_EQ(x.b, 6.0);
}

TEST(DualNumberTests, DualNumberAxpy) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y(3.0, 4.0);
    mathlib::axpy(2.0, x, y);
    EXPECT_DOUBLE_EQ(y.a, 5.0);
    EXPECT_DOUBLE_EQ(y.b, 8.0);
}

TEST(DualNumberTests, HyperDualNumberAddition) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    mathlib::HyperDual2<double> z = x + y;
    EXPECT_DOUBLE_EQ(z.a, 5.0);
    EXPECT_DOUBLE_EQ(z.b, 7.0);
    EXPECT_DOUBLE_EQ(z.c, 9.0);
    EXPECT_DOUBLE_EQ(z.d, 11.0);
}


TEST(DualNumberTests, HyperDualNumberSubtraction) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    mathlib::HyperDual2<double> z = x - y;
    EXPECT_DOUBLE_EQ(z.a, -3.0);
    EXPECT_DOUBLE_EQ(z.b, -3.0);
    EXPECT_DOUBLE_EQ(z.c, -3.0);
}

TEST(DualNumberTests, HyperDualNumberInversion) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x.inv();
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, -2.0);
    EXPECT_DOUBLE_EQ(y.c, -3.0);
    EXPECT_DOUBLE_EQ(y.d, 12.0 - 4.0);
}

TEST(DualNumberTests, HyperDualNumberMulDiv) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    auto y = x.inv();
    auto z = x * y;
    EXPECT_DOUBLE_EQ(z.a, 1.0);
    EXPECT_DOUBLE_EQ(z.b, 0.0);
    EXPECT_DOUBLE_EQ(z.c, 0.0);
    EXPECT_DOUBLE_EQ(z.d, 0.0);
}

TEST(DualNumberTests, HyperDualNumberScalarMultiplication) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = 2.0 * x;
    EXPECT_DOUBLE_EQ(y.a, 2.0);
    EXPECT_DOUBLE_EQ(y.b, 4.0);
    EXPECT_DOUBLE_EQ(y.c, 6.0);
}

TEST(DualNumberTests, HyperDualNumberScalarDivision) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x / 2.0;
    EXPECT_DOUBLE_EQ(y.a, 0.5);
    EXPECT_DOUBLE_EQ(y.b, 1.0);
    EXPECT_DOUBLE_EQ(y.c, 1.5);
}

TEST(DualNumberTests, HyperDualNumberUnaryMinus) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = -x;
    EXPECT_DOUBLE_EQ(y.a, -1.0);
    EXPECT_DOUBLE_EQ(y.b, -2.0);
    EXPECT_DOUBLE_EQ(y.c, -3.0);
    EXPECT_DOUBLE_EQ(y.d, -4.0);
}

TEST(DualNumberTests, HyperDualNumberTan) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = tan(x);
    EXPECT_DOUBLE_EQ(y.a, std::tan(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (std::cos(1.0) * std::cos(1.0)));
    EXPECT_DOUBLE_EQ(y.c, 3.0 / (std::cos(1.0) * std::cos(1.0)));
    EXPECT_DOUBLE_EQ(y.d, 4.0 / (std::cos(1.0) * std::cos(1.0)) + 6.0 * 2.0 * std::tan(1.0) / (std::cos(1.0) * std::cos(1.0)));
}

TEST(DualNumberTests, HyperDualNumberSinh) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = sinh(x);
    EXPECT_DOUBLE_EQ(y.a, std::sinh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 * std::cosh(1.0));
    EXPECT_DOUBLE_EQ(y.c, 3.0 * std::cosh(1.0));
    EXPECT_DOUBLE_EQ(y.d, 4.0 * std::cosh(1.0) + 6.0 * std::sinh(1.0));
}

TEST(DualNumberTests, HyperDualNumberCosh) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = cosh(x);
    EXPECT_DOUBLE_EQ(y.a, std::cosh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 * std::sinh(1.0));
    EXPECT_DOUBLE_EQ(y.c, 3.0 * std::sinh(1.0));
    EXPECT_DOUBLE_EQ(y.d, 4.0 * std::sinh(1.0) + 6.0 * std::cosh(1.0));
}

TEST(DualNumberTests, HyperDualNumberTanh) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = tanh(x);
    EXPECT_DOUBLE_EQ(y.a, std::tanh(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (std::cosh(1.0) * std::cosh(1.0)));
    EXPECT_DOUBLE_EQ(y.c, 3.0 / (std::cosh(1.0) * std::cosh(1.0)));
    EXPECT_DOUBLE_EQ(y.d, 4.0 / (std::cosh(1.0) * std::cosh(1.0)) + 6.0 * 2.0 * std::tanh(1.0) / (std::cosh(1.0) * std::cosh(1.0)));
}

TEST(DualNumberTests, HyperDualNumberPowScalar) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = pow(x, 2.0);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0 * 2.0 / 1.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0 * 2.0 / 1.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0 * 2.0 / 1.0 + 6.0 * 2.0);
}

TEST(DualNumberTests, HyperDualNumberSqrt) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = sqrt(x);
    EXPECT_DOUBLE_EQ(y.a, std::sqrt(1.0));
    EXPECT_DOUBLE_EQ(y.b, 2.0 / (2.0 * std::sqrt(1.0)));
    EXPECT_DOUBLE_EQ(y.c, 3.0 / (2.0 * std::sqrt(1.0)));
    EXPECT_DOUBLE_EQ(y.d, 4.0 / (2.0 * std::sqrt(1.0)) - 6.0 / (4.0 * std::pow(1.0, 3.0/2.0)));
}

TEST(DualNumberTests, HyperDualNumberComparison) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> z(2.0, 3.0, 4.0, 5.0);
    EXPECT_TRUE(x == y);
    EXPECT_FALSE(x == z);
    EXPECT_FALSE(x != y);
    EXPECT_TRUE(x != z);
}

TEST(DualNumberTests, HyperDualNumberCopy) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(x);
    EXPECT_DOUBLE_EQ(x.a, y.a);
    EXPECT_DOUBLE_EQ(x.b, y.b);
    EXPECT_DOUBLE_EQ(x.c, y.c);
}

TEST(DualNumberTests, HyperDualNumberMove) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(std::move(x));
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0);
}

TEST(DualNumberTests, HyperDualNumberAssignment) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y;
    y = x;
    EXPECT_DOUBLE_EQ(x.a, y.a);
    EXPECT_DOUBLE_EQ(x.b, y.b);
    EXPECT_DOUBLE_EQ(x.c, y.c);
    EXPECT_DOUBLE_EQ(x.d, y.d);
}

TEST(DualNumberTests, HyperDualNumberAssignmentMove) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y;
    y = std::move(x);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberAssignementAddition) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    x += y;
    EXPECT_DOUBLE_EQ(x.a, 5.0);
    EXPECT_DOUBLE_EQ(x.b, 7.0);
    EXPECT_DOUBLE_EQ(x.c, 9.0);
    EXPECT_DOUBLE_EQ(x.d, 11.0);
}

TEST(DualNumberTests, HyperDualNumberAssignementSubtraction) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    x -= y;
    EXPECT_DOUBLE_EQ(x.a, -3.0);
    EXPECT_DOUBLE_EQ(x.b, -3.0);
    EXPECT_DOUBLE_EQ(x.c, -3.0);
    EXPECT_DOUBLE_EQ(x.d, -3.0);
}

TEST(DualNumberTests, HyperDualNumberAssignmentMultiplication) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    x *= y;
    EXPECT_DOUBLE_EQ(x.a, 4.0);
    EXPECT_DOUBLE_EQ(x.b, 13.0);
    EXPECT_DOUBLE_EQ(x.c, 18.0);
    EXPECT_DOUBLE_EQ(x.d, 50.0);
}

TEST(DualNumberTests, HyperDualNumberAssignmentScalarAddition) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    x += 3.0;
    EXPECT_DOUBLE_EQ(x.a, 4.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0);
    EXPECT_DOUBLE_EQ(x.c, 3.0);
    EXPECT_DOUBLE_EQ(x.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberAssignmentScalarSubtraction) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    x -= 3.0;
    EXPECT_DOUBLE_EQ(x.a, -2.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0);
    EXPECT_DOUBLE_EQ(x.c, 3.0);
    EXPECT_DOUBLE_EQ(x.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberAssignmentScalarMultiplication) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    x *= 3.0;
    EXPECT_DOUBLE_EQ(x.a, 3.0);
    EXPECT_DOUBLE_EQ(x.b, 6.0);
    EXPECT_DOUBLE_EQ(x.c, 9.0);
    EXPECT_DOUBLE_EQ(x.d, 12.0);
}

TEST(DualNumberTests, HyperDualNumberAssignmentScalarDivision) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    x /= 3.0;
    EXPECT_DOUBLE_EQ(x.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.b, 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.c, 3.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.d, 4.0 / 3.0);
}

TEST(DualNumberTests, HyperDualNumberAdditionScalar) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x + 3.0;
    EXPECT_DOUBLE_EQ(y.a, 4.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberSubtractionScalar) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x - 3.0;
    EXPECT_DOUBLE_EQ(y.a, -2.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberMultiplicationScalar) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x * 3.0;
    EXPECT_DOUBLE_EQ(y.a, 3.0);
    EXPECT_DOUBLE_EQ(y.b, 6.0);
    EXPECT_DOUBLE_EQ(y.c, 9.0);
    EXPECT_DOUBLE_EQ(y.d, 12.0);
}

TEST(DualNumberTests, HyperDualNumberDivisionScalar) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = x / 3.0;
    EXPECT_DOUBLE_EQ(y.a, 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0 / 3.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0 / 3.0);
}

TEST(DualNumberTests, HyperDualNumberAdditionScalarReverse) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = 3.0 + x;
    EXPECT_DOUBLE_EQ(y.a, 4.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
    EXPECT_DOUBLE_EQ(y.c, 3.0);
    EXPECT_DOUBLE_EQ(y.d, 4.0);
}

TEST(DualNumberTests, HyperDualNumberSubtractionScalarReverse) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = 3.0 - x;
    EXPECT_DOUBLE_EQ(y.a, 2.0);
    EXPECT_DOUBLE_EQ(y.b, -2.0);
    EXPECT_DOUBLE_EQ(y.c, -3.0);
    EXPECT_DOUBLE_EQ(y.d, -4.0);
}

TEST(DualNumberTests, HyperDualNumberMultiplicationScalarReverse) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y = 3.0 * x;
    EXPECT_DOUBLE_EQ(y.a, 3.0);
    EXPECT_DOUBLE_EQ(y.b, 6.0);
    EXPECT_DOUBLE_EQ(y.c, 9.0);
    EXPECT_DOUBLE_EQ(y.d, 12.0);
}

TEST(DualNumberTests, HyperDualNumberScale) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::scale(3.0, x);
    EXPECT_DOUBLE_EQ(x.a, 3.0);
    EXPECT_DOUBLE_EQ(x.b, 6.0);
    EXPECT_DOUBLE_EQ(x.c, 9.0);
    EXPECT_DOUBLE_EQ(x.d, 12.0);
}

TEST(DualNumberTests, HyperDualNumberAxpy) {
    mathlib::HyperDual2<double> x(1.0, 2.0, 3.0, 4.0);
    mathlib::HyperDual2<double> y(4.0, 5.0, 6.0, 7.0);
    mathlib::axpy(2.0, x, y);
    EXPECT_DOUBLE_EQ(y.a, 6.0);
    EXPECT_DOUBLE_EQ(y.b, 9.0);
    EXPECT_DOUBLE_EQ(y.c, 12.0);
    EXPECT_DOUBLE_EQ(y.d, 15.0);
}
