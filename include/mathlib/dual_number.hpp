#ifndef MATHLIB_DUAL_NUMBER_HPP
#define MATHLIB_DUAL_NUMBER_HPP

#include <cmath>
#include <concepts>

namespace mathlib {

template<std::floating_point T>
struct Dual {
    T a;
    T b;

    Dual() noexcept: a(0.0), b(0.0) {}
    Dual(T a, T b) noexcept: a(a), b(b) {}


    auto operator+=(const Dual<T>& x) noexcept -> Dual<T>& {
        this->a += x.a;
        this->b += x.b;
        return *this;
    }

    auto operator-=(const Dual<T>& x) noexcept -> Dual<T>& {
        this->a -= x.a;
        this->b -= x.b;
        return *this;
    }

    auto operator*=(const Dual<T>& x) noexcept -> Dual<T>& {
        T a = this->a;
        this->a = a * x.a;
        this->b = a * x.b + this->b * x.a;
        return *this;
    }

    auto operator/=(const Dual<T>& x) noexcept -> Dual<T>& {
        T a = this->a;
        this->a = a / x.a;
        this->b = (this->b * x.a - a * x.b) / (x.a * x.a);
        return *this;
    }

    auto operator+=(T alpha) noexcept -> Dual<T>& {
        this->a += alpha;
        return *this;
    }

    auto operator-=(T alpha) noexcept -> Dual<T>& {
        this->a -= alpha;
        return *this;
    }

    auto operator*=(T alpha) noexcept -> Dual<T>& {
        this->a *= alpha;
        this->b *= alpha;
        return *this;
    }

    auto operator/=(T alpha) noexcept -> Dual<T>& {
        this->a /= alpha;
        this->b /= alpha;
        return *this;
    }
};

// ==== Implementation === //

// ======================= //
// Comparison operators    //
// ======================= //

/// @brief  Equality operator for dual numbers.
template<std::floating_point T>
constexpr auto operator==(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> bool {
    return lhs.a == rhs.a && lhs.b == rhs.b;
}

/// @brief  Inequality operator for dual numbers.
template<std::floating_point T>
constexpr auto operator!=(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> bool {
    return lhs.a != rhs.a || lhs.b != rhs.b;
}


// ======================= // 
// Arithmetic operators    //
// ======================= //

/// @brief  Negation operator for dual numbers.
template<std::floating_point T>
auto operator-(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(-x.a, -x.b);
}

/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(lhs.a + rhs.a, lhs.b + rhs.b);
}

/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(const Dual<T>& lhs, T alpha) noexcept -> Dual<T> {
    return Dual<T>(lhs.a + alpha, lhs.b);
}

/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(T alpha, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(alpha + rhs.a, rhs.b);
}


/// @brief  Subtraction operator for dual numbers. 
template<std::floating_point T>
constexpr auto operator-(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(lhs.a - rhs.a, lhs.b - rhs.b);
}

/// @brief  Subtraction operator for dual numbers.
template<std::floating_point T>
constexpr auto operator-(const Dual<T>& lhs, T alpha) noexcept -> Dual<T> {
    return Dual<T>(lhs.a - alpha, lhs.b);
}

/// @brief  Subtraction operator for dual numbers.
template<std::floating_point T>
constexpr auto operator-(T alpha, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(alpha - rhs.a, -rhs.b);
}


/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(lhs.a * rhs.a, lhs.a * rhs.b + lhs.b * rhs.a);
}

/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(const Dual<T>& lhs, T alpha) noexcept -> Dual<T> {
    return Dual<T>(lhs.a * alpha, lhs.b * alpha);
}

/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(T alpha, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(alpha * rhs.a, alpha * rhs.b);
}


/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(const Dual<T>& lhs, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(lhs.a / rhs.a, (lhs.b * rhs.a - lhs.a * rhs.b) / (rhs.a * rhs.a));
}

/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(const Dual<T>& lhs, T alpha) noexcept -> Dual<T> {
    return Dual<T>(lhs.a / alpha, lhs.b / alpha);
}

/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(T alpha, const Dual<T>& rhs) noexcept -> Dual<T> {
    return Dual<T>(alpha / rhs.a, -alpha * rhs.b / (rhs.a * rhs.a));
}


/// @brief  Scalar multiplication for dual numbers.
template<std::floating_point T>
constexpr auto scale(T alpha, Dual<T>& x) noexcept -> Dual<T>& {
    x.a *= alpha;
    x.b *= alpha;
    return x;
}

/// @brief  AXPY operation for dual numbers. (y = alpha * x + y)
template<std::floating_point T>
constexpr auto axpy(T alpha, const Dual<T>& x, Dual<T>& y) noexcept -> Dual<T>& {
    y.a += alpha * x.a; 
    y.b += alpha * x.b;
    return y;
}


// ======================= //
// Elementary functions    //
// ======================= //

/// @brief  Exponential function for dual numbers.
template<std::floating_point T>
constexpr auto exp(const Dual<T>& x) noexcept -> Dual<T> {
    T exp_a = std::exp(x.a);
    return Dual<T>(exp_a, x.b * exp_a);
}

/// @brief  Logarithm function for dual numbers.
template<std::floating_point T>
constexpr auto log(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::log(x.a), x.b / x.a);
}

/// @brief  Sine function for dual numbers.
template<std::floating_point T>
constexpr auto sin(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::sin(x.a), x.b * std::cos(x.a));
}

/// @brief  Cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cos(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::cos(x.a), -x.b * std::sin(x.a));
}

/// @brief  Tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tan(const Dual<T>& x) noexcept -> Dual<T> {
    T cos_a = std::cos(x.a);
    return Dual<T>(std::tan(x.a), x.b / (cos_a * cos_a));
}

/// @brief  Hyperbolic sine function for dual numbers.
template<std::floating_point T>
constexpr auto sinh(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::sinh(x.a), x.b * std::cosh(x.a));
}

/// @brief  Hyperbolic cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cosh(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::cosh(x.a), x.b * std::sinh(x.a));
}

/// @brief  Hyperbolic tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tanh(const Dual<T>& x) noexcept -> Dual<T> {
    T cosh_a = std::cosh(x.a);
    return Dual<T>(std::tanh(x.a), x.b / (cosh_a * cosh_a));
}

/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(const Dual<T>& x, T alpha) noexcept -> Dual<T> {
    T pow_a = std::pow(x.a, alpha);
    return Dual<T>(pow_a, alpha * pow_a * x.b / x.a);
}

/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(const Dual<T>& x, const Dual<T>& alpha) noexcept -> Dual<T> {
    T pow_a = std::pow(x.a, alpha.a);
    return Dual<T>(pow_a, pow_a * (alpha.b * std::log(x.a) + alpha.a * x.b / x.a));
}

/// @brief  Square root function for dual numbers.
template<std::floating_point T>
auto sqrt(const Dual<T>& x) noexcept -> Dual<T> {
    T sqrt_a = std::sqrt(x.a);
    return Dual<T>(sqrt_a, x.b / (2.0 * sqrt_a));
}

/// @brief  Absolute value function for dual numbers.
template<std::floating_point T>
auto abs(const Dual<T>& x) noexcept -> Dual<T> {
    return x.a >= 0.0 ? x : -x;
}

/// @brief  Sign function for dual numbers.
template<std::floating_point T>
auto sign(const Dual<T>& x) noexcept -> Dual<T> {
    return x.a > 0.0 ? Dual<T>(1.0, 0.0) : x.a < 0.0 ? Dual<T>(-1.0, 0.0) : Dual<T>(0.0, 0.0);
}

}

#endif