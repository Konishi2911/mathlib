#include <gtest/gtest.h>
#include <random>
#include "mathlib/nlp/cost_function.hpp"
#include "lalib/mat.hpp"
#include "lalib/vec.hpp"

auto quadrature_func(double x) noexcept -> double {
    return (x - 2.0)*(x + 1.0);
}
auto deriv_quadrature_func(double x) noexcept -> double {
    return 2.0 * x - 1.0;
}

auto rosenbrock_func(const lalib::DynVecD& x) noexcept -> double {
    auto n = x.size();
    auto r = 0.0;
    for (auto i = 0u; i < n - 1; ++i) {
        r += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1.0 - x[i], 2);
    }
    return r;
}
auto deriv_rosenbrock_func(const lalib::DynVecD& x) noexcept -> lalib::DynVecD {
    auto n = x.size();
    auto r = lalib::DynVecD::filled(n, 0.0);
    r[0] = -400.0 * (x[1] - x[0]*x[0]) * x[0] - 2.0 * (1.0 - x[0]);
    for (auto i = 1u; i < n - 1; ++i) {
        r[i] = 200.0 * (x[i] - x[i-1]*x[i-1]) - 400.0 * (x[i+1] - x[i]*x[i]) * x[i] - 2.0 * (1.0 - x[i]);
    }
    r[n - 1] = 200.0 * (x[n-1] - x[n-2]*x[n-2]);
    return r;
}
auto hess_rosenbrock_func(const lalib::DynVecD& x) noexcept -> lalib::DynMatD {
    auto n = x.size();
    auto r = std::vector<double>(n * n, 0.0);
    r[0] = 1200.0 * x[0]*x[0] - 400 * x[1] + 2;
    r[1] = -400.0 * x[0];
    for (auto i = 1u; i < n - 1; ++i) {
        r[i * n + i - 1] = -400.0 * x[i - 1];
        r[i * n + i] = 1200.0 * x[i]*x[i] - 400 * x[i + 1] + 202;
        r[i * n + i + 1] = -400 * x[i];
    }
    r[n * n - 2] = -400 * x[n - 2];
    r[n * n - 1] = 200;
    return lalib::DynMatD(std::move(r), n, n);
}

TEST(NumericCostFuncTests, DoubleGradTest) {
    auto mt = std::mt19937(std::random_device()());
    auto rng = std::uniform_real_distribution<double>(-10, 10);
    auto func = mathlib::nlp::NumericCostFunc<double, double(*)(double)>(quadrature_func, 1e-5);

    for (auto i = 0u; i < 10; ++i) {
        auto x = rng(mt);
        EXPECT_NEAR(deriv_quadrature_func(x), func.grad(x), 1e-5);
    }
}

TEST(NumericCostFuncTests, DoubleHessTest) {
    auto mt = std::mt19937(std::random_device()());
    auto rng = std::uniform_real_distribution<double>(-10, 10);
    auto func = mathlib::nlp::NumericCostFunc<double, double(*)(double)>(quadrature_func, 1e-5);

    for (auto i = 0u; i < 10; ++i) {
        auto x = rng(mt);
        EXPECT_NEAR(2.0, func.hessian(x), 1e-3);
    }
}

TEST(NumericCostFuncTests, VecGradTest) {
    auto mt = std::mt19937(std::random_device()());
    auto rng = std::uniform_real_distribution<double>(-5, 5);
    auto func = mathlib::nlp::NumericCostFunc<lalib::DynVecD, double(*)(const lalib::DynVecD&)>(rosenbrock_func, 1e-5);

    for (auto i = 0u; i < 10; ++i) {
        auto x = lalib::DynVecD::filled(5, rng(mt));
        EXPECT_NEAR(deriv_rosenbrock_func(x)[0], func.grad(x)[0], 1e-4);
        EXPECT_NEAR(deriv_rosenbrock_func(x)[1], func.grad(x)[1], 1e-4);
        EXPECT_NEAR(deriv_rosenbrock_func(x)[2], func.grad(x)[2], 1e-4);
        EXPECT_NEAR(deriv_rosenbrock_func(x)[3], func.grad(x)[3], 1e-4);
        EXPECT_NEAR(deriv_rosenbrock_func(x)[4], func.grad(x)[4], 1e-4);
    }
}

TEST(NumericCostFuncTests, VecHessTest) {
    auto mt = std::mt19937(std::random_device()());
    auto rng = std::uniform_real_distribution<double>(-5, 5);
    auto func = mathlib::nlp::NumericCostFunc<lalib::DynVecD, double(*)(const lalib::DynVecD&)>(rosenbrock_func, 1e-4);

    for (auto i = 0u; i < 10; ++i) {
        auto x = lalib::DynVecD::filled(4, rng(mt));
        EXPECT_NEAR(hess_rosenbrock_func(x)(0, 0), func.hessian(x)(0, 0), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(0, 1), func.hessian(x)(0, 1), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(0, 2), func.hessian(x)(0, 2), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(0, 3), func.hessian(x)(0, 3), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(1, 0), func.hessian(x)(1, 0), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(1, 1), func.hessian(x)(1, 1), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(1, 2), func.hessian(x)(1, 2), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(1, 3), func.hessian(x)(1, 3), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(2, 0), func.hessian(x)(2, 0), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(2, 1), func.hessian(x)(2, 1), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(2, 2), func.hessian(x)(2, 2), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(2, 3), func.hessian(x)(2, 3), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(3, 0), func.hessian(x)(3, 0), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(3, 1), func.hessian(x)(3, 1), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(3, 2), func.hessian(x)(3, 2), 1e-2);
        EXPECT_NEAR(hess_rosenbrock_func(x)(3, 3), func.hessian(x)(3, 3), 1e-2);
    }
}