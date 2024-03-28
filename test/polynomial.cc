#include <gtest/gtest.h>
#include "lalib/vec.hpp"
#include "mathlib/polynomial.hpp"


TEST(PolynomialTests, DoublePointTest) {
    // quadratic function
    auto polynomial = mathlib::Polynomial<double>(std::vector { 0.0, 0.0, 1.0 });

    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0));
    ASSERT_DOUBLE_EQ(0.25, polynomial(0.5));
    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0));
}

TEST(PolynomialTests, DoubleDerivationTest) {
    // quadratic function
    auto polynomial = mathlib::Polynomial<double>(std::vector { 0.0, 0.0, 1.0 });

    ASSERT_DOUBLE_EQ(0.0, polynomial.deriv(0.0));
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5));
    ASSERT_DOUBLE_EQ(2.0, polynomial.deriv(1.0));
}

TEST(PolynomialTests, SizedVecPointTest) {
    // quadratic function
    auto polynomial = mathlib::Polynomial<lalib::VecD<3>>(std::vector {
        lalib::VecD<3>({ 0.0, 0.0, 0.0 }),
        lalib::VecD<3>({ 1.0, 1.0, 0.0 }),
        lalib::VecD<3>({ 0.0, 0.0, 1.0 })
    });

    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[0]);
    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[1]);
    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[2]);

    ASSERT_DOUBLE_EQ(0.5, polynomial(0.5)[0]);
    ASSERT_DOUBLE_EQ(0.5, polynomial(0.5)[1]);
    ASSERT_DOUBLE_EQ(0.25, polynomial(0.5)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[1]);
    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[2]);
}

TEST(PolynomialTests, SizedVecDerivationTest) {
    // Pure quadratic function
    auto polynomial = mathlib::Polynomial<lalib::VecD<3>>(std::vector {
        lalib::VecD<3>({ 0.0, 0.0, 0.0 }),
        lalib::VecD<3>({ 1.0, 1.0, 0.0 }),
        lalib::VecD<3>({ 0.0, 0.0, 1.0 })
    });

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.0)[1]);
    ASSERT_DOUBLE_EQ(0.0, polynomial.deriv(0.0)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[1]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(1.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(1.0)[1]);
    ASSERT_DOUBLE_EQ(2.0, polynomial.deriv(1.0)[2]);
}

TEST(PolynomialTests, DynVecPointTest) {
    // quadratic function
    auto polynomial = mathlib::Polynomial<lalib::DynVecD>(std::vector {
        lalib::DynVecD({ 0.0, 0.0, 0.0 }),
        lalib::DynVecD({ 1.0, 1.0, 0.0 }),
        lalib::DynVecD({ 0.0, 0.0, 1.0 })
    });

    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[0]);
    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[1]);
    ASSERT_DOUBLE_EQ(0.0, polynomial(0.0)[2]);

    ASSERT_DOUBLE_EQ(0.5, polynomial(0.5)[0]);
    ASSERT_DOUBLE_EQ(0.5, polynomial(0.5)[1]);
    ASSERT_DOUBLE_EQ(0.25, polynomial(0.5)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[1]);
    ASSERT_DOUBLE_EQ(1.0, polynomial(1.0)[2]);
}

TEST(PolynomialTests, DynVecDerivationTest) {
    // Pure quadratic function
    auto polynomial = mathlib::Polynomial<lalib::DynVec<double>>(std::vector {
        lalib::DynVec<double>({ 0, 0, 0 }),
        lalib::DynVec<double>({ 1, 1, 0 }),
        lalib::DynVec<double>({ 0, 0, 1 })
    });

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.0)[1]);
    ASSERT_DOUBLE_EQ(0.0, polynomial.deriv(0.0)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[1]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(0.5)[2]);

    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(1.0)[0]);
    ASSERT_DOUBLE_EQ(1.0, polynomial.deriv(1.0)[1]);
    ASSERT_DOUBLE_EQ(2.0, polynomial.deriv(1.0)[2]);
}