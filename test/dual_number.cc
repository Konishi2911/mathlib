#include "mathlib/dual_number.hpp"
#include <gtest/gtest.h>

template<typename T>
auto f1(T x) -> T { return x * sin(x); }

template<typename T>
auto df1(T x) -> T { return sin(x) + x * cos(x); }


TEST(DualNumberTests, DerivationTest1) {
    mathlib::Dual<double> x(1.0, 1.0);
    mathlib::Dual<double> y = f1(x);
    EXPECT_DOUBLE_EQ(y.a, f1(1.0));
    EXPECT_DOUBLE_EQ(y.b, df1(1.0));
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

TEST(DualNumberTests, DualNumberAbs) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = abs(x);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 2.0);
}

TEST(DualNumberTests, DualNumberSign) {
    mathlib::Dual<double> x(1.0, 2.0);
    mathlib::Dual<double> y = sign(x);
    EXPECT_DOUBLE_EQ(y.a, 1.0);
    EXPECT_DOUBLE_EQ(y.b, 0.0);
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
