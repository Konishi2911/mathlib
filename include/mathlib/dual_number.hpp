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

template<std::floating_point T>
struct HyperDual2 {
    T a;
    T b;
    T c;
    T d;

    HyperDual2() noexcept: a(0.0), b(0.0), c(0.0), d(0.0) {}
    HyperDual2(T a, T b, T c, T d) noexcept: a(a), b(b), c(c), d(d) {}

    auto inv() const noexcept -> HyperDual2<T> {
        T inv_a = 1.0 / this->a;
        T inv_b = -this->b * inv_a * inv_a;
        T inv_c = -this->c * inv_a * inv_a;
        T inv_d = 2.0 * this->b * this->c * inv_a * inv_a * inv_a - this->d * inv_a * inv_a;

        return HyperDual2<T>(inv_a, inv_b, inv_c, inv_d);
    }

    auto operator+=(const HyperDual2<T>& x) noexcept -> HyperDual2<T>& {
        this->a += x.a;
        this->b += x.b;
        this->c += x.c;
        this->d += x.d;
        return *this;
    }

    auto operator-=(const HyperDual2<T>& x) noexcept -> HyperDual2<T>& {
        this->a -= x.a;
        this->b -= x.b;
        this->c -= x.c;
        this->d -= x.d;
        return *this;
    }

    auto operator*=(const HyperDual2<T>& x) noexcept -> HyperDual2<T>& {
        T a = this->a;
        T b = this->b;
        T c = this->c;
        this->a = a * x.a;
        this->b = a * x.b + b * x.a;
        this->c = a * x.c + c * x.a;
        this->d = a * x.d + this->d * x.a + b * x.c + c * x.b;
        return *this;
    }

    auto operator/=(const HyperDual2<T>& x) noexcept -> HyperDual2<T>& {
        HyperDual2<T> inv_x = x.inv();
        return *this *= inv_x;
    }

    auto operator+=(T alpha) noexcept -> HyperDual2<T>& {
        this->a += alpha;
        return *this;
    }

    auto operator-=(T alpha) noexcept -> HyperDual2<T>& {
        this->a -= alpha;
        return *this;
    }

    auto operator*=(T alpha) noexcept -> HyperDual2<T>& {
        this->a *= alpha;
        this->b *= alpha;
        this->c *= alpha;
        this->d *= alpha;
        return *this;
    }

    auto operator/=(T alpha) noexcept -> HyperDual2<T>& {
        this->a /= alpha;
        this->b /= alpha;
        this->c /= alpha;
        this->d /= alpha;
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

/// @brief  Equality operator for dual numbers.
template<std::floating_point T>
constexpr auto operator==(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> bool {
    return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c;
}

/// @brief  Inequality operator for dual numbers.
template<std::floating_point T>
constexpr auto operator!=(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> bool {
    return lhs.a != rhs.a || lhs.b != rhs.b || lhs.c != rhs.c;
}


// ======================= // 
// Arithmetic operators    //
// ======================= //

/// @brief  Negation operator for dual numbers.
template<std::floating_point T>
auto operator-(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(-x.a, -x.b);
}

/// @brief  Negation operator for dual numbers.
template<std::floating_point T>
auto operator-(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(-x.a, -x.b, -x.c, -x.d);
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


/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a + rhs.a, lhs.b + rhs.b, lhs.c + rhs.c, lhs.d + rhs.d);
}

/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(const HyperDual2<T>& lhs, T alpha) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a + alpha, lhs.b, lhs.c, lhs.d);
}

/// @brief  Addition operator for dual numbers.
template<std::floating_point T>
constexpr auto operator+(T alpha, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(alpha + rhs.a, rhs.b, rhs.c, rhs.d);
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


/// @brief  Subtraction operator for dual numbers. 
template<std::floating_point T>
constexpr auto operator-(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a - rhs.a, lhs.b - rhs.b, lhs.c - rhs.c, lhs.d - rhs.d);
}

/// @brief  Subtraction operator for dual numbers.
template<std::floating_point T>
constexpr auto operator-(const HyperDual2<T>& lhs, T alpha) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a - alpha, lhs.b, lhs.c, lhs.d);
}

/// @brief  Subtraction operator for dual numbers.
template<std::floating_point T>
constexpr auto operator-(T alpha, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(alpha - rhs.a, -rhs.b, -rhs.c, -rhs.d);
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


/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        lhs.a * rhs.a, 
        lhs.a * rhs.b + lhs.b * rhs.a, 
        lhs.a * rhs.c + lhs.c * rhs.a,
        lhs.a * rhs.d + lhs.d * rhs.a + lhs.b * rhs.c + lhs.c * rhs.b
    );
}

/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(const HyperDual2<T>& lhs, T alpha) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a * alpha, lhs.b * alpha, lhs.c * alpha, lhs.d * alpha);
}

/// @brief  Multiplication operator for dual numbers.
template<std::floating_point T>
constexpr auto operator*(T alpha, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    return rhs * alpha;
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


/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(const HyperDual2<T>& lhs, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    HyperDual2<T> inv_rhs = rhs.inv();
    return lhs * inv_rhs;
}

/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(const HyperDual2<T>& lhs, T alpha) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(lhs.a / alpha, lhs.b / alpha, lhs.c / alpha, lhs.d / alpha);
}

/// @brief  Division operator for dual numbers.
template<std::floating_point T>
constexpr auto operator/(T alpha, const HyperDual2<T>& rhs) noexcept -> HyperDual2<T> {
    HyperDual2<T> inv_rhs = rhs.inv();
    return alpha * inv_rhs;
}


/// @brief  Scalar multiplication for dual numbers.
template<std::floating_point T>
constexpr auto scale(T alpha, Dual<T>& x) noexcept -> Dual<T>& {
    x.a *= alpha;
    x.b *= alpha;
    return x;
}

/// @brief  Scalar multiplication for dual numbers.
template<std::floating_point T>
constexpr auto scale(T alpha, HyperDual2<T>& x) noexcept -> HyperDual2<T>& {
    x.a *= alpha;
    x.b *= alpha;
    x.c *= alpha;
    x.d *= alpha;
    return x;
}

/// @brief  AXPY operation for dual numbers. (y = alpha * x + y)
template<std::floating_point T>
constexpr auto axpy(T alpha, const Dual<T>& x, Dual<T>& y) noexcept -> Dual<T>& {
    y.a += alpha * x.a; 
    y.b += alpha * x.b;
    return y;
}

/// @brief  AXPY operation for dual numbers. (y = alpha * x + y)
template<std::floating_point T>
constexpr auto axpy(T alpha, const HyperDual2<T>& x, HyperDual2<T>& y) noexcept -> HyperDual2<T>& {
    y.a += alpha * x.a; 
    y.b += alpha * x.b;
    y.c += alpha * x.c;
    y.d += alpha * x.d;
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

/// @brief  Exponential function for dual numbers.
template<std::floating_point T>
constexpr auto exp(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    T exp_a = std::exp(x.a);
    return HyperDual2<T>(exp_a, x.b * exp_a, x.c * exp_a, x.d * exp_a + x.b * x.c * exp_a);
}


/// @brief  Logarithm function for dual numbers.
template<std::floating_point T>
constexpr auto log(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::log(x.a), x.b / x.a);
}

/// @brief  Logarithm function for dual numbers.
template<std::floating_point T>
constexpr auto log(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        std::log(x.a), 
        x.b / x.a, 
        x.c / x.a, 
        x.d / x.a - x.b * x.c / (x.a * x.a)
    );
}


/// @brief  Sine function for dual numbers.
template<std::floating_point T>
constexpr auto sin(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::sin(x.a), x.b * std::cos(x.a));
}

/// @brief  Sine function for dual numbers.
template<std::floating_point T>
constexpr auto sin(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        std::sin(x.a), 
        x.b * std::cos(x.a), 
        x.c * std::cos(x.a), 
        x.d * std::cos(x.a) - x.b * x.c * std::sin(x.a)
    );
}


/// @brief  Cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cos(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::cos(x.a), -x.b * std::sin(x.a), -x.b * std::sin(x.a));
}

/// @brief  Cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cos(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        std::cos(x.a), 
        -x.b * std::sin(x.a), 
        -x.c * std::sin(x.a), 
        -x.d * std::sin(x.a) - x.b * x.c * std::cos(x.a)
    );
}


/// @brief  Tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tan(const Dual<T>& x) noexcept -> Dual<T> {
    T cos_a = std::cos(x.a);
    return Dual<T>(std::tan(x.a), x.b / (cos_a * cos_a));
}

/// @brief  Tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tan(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    T cos_a = std::cos(x.a);
    return HyperDual2<T>(
        std::tan(x.a), 
        x.b / (cos_a * cos_a), 
        x.c / (cos_a * cos_a), 
        x.d / (cos_a * cos_a) + 2.0 * x.b * x.c * std::tan(x.a) / (cos_a * cos_a)
    );
}


/// @brief  Hyperbolic sine function for dual numbers.
template<std::floating_point T>
constexpr auto sinh(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::sinh(x.a), x.b * std::cosh(x.a));
}

/// @brief  Hyperbolic sine function for dual numbers.
template<std::floating_point T>
constexpr auto sinh(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        std::sinh(x.a), 
        x.b * std::cosh(x.a), 
        x.c * std::cosh(x.a), 
        x.d * std::cosh(x.a) + x.b * x.c * std::sinh(x.a)
    );
}


/// @brief  Hyperbolic cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cosh(const Dual<T>& x) noexcept -> Dual<T> {
    return Dual<T>(std::cosh(x.a), x.b * std::sinh(x.a));
}

/// @brief  Hyperbolic cosine function for dual numbers.
template<std::floating_point T>
constexpr auto cosh(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    return HyperDual2<T>(
        std::cosh(x.a), 
        x.b * std::sinh(x.a), 
        x.c * std::sinh(x.a), 
        x.d * std::sinh(x.a) + x.b * x.c * std::cosh(x.a)
    );
}


/// @brief  Hyperbolic tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tanh(const Dual<T>& x) noexcept -> Dual<T> {
    T cosh_a = std::cosh(x.a);
    return Dual<T>(std::tanh(x.a), x.b / (cosh_a * cosh_a));
}

/// @brief  Hyperbolic tangent function for dual numbers.
template<std::floating_point T>
constexpr auto tanh(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    T cosh_a = std::cosh(x.a);
    return HyperDual2<T>(
        std::tanh(x.a), 
        x.b / (cosh_a * cosh_a), 
        x.c / (cosh_a * cosh_a),
        x.d / (cosh_a * cosh_a) + 2.0 * x.b * x.c * std::tanh(x.a) / (cosh_a * cosh_a)
    );
}


/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(const Dual<T>& x, T alpha) noexcept -> Dual<T> {
    T pow_a = std::pow(x.a, alpha);
    return Dual<T>(pow_a, alpha * pow_a * x.b / x.a);
}

/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(T alpha, const Dual<T>& x) noexcept -> Dual<T> {
    T pow_a = std::pow(alpha, x.a);
    return Dual<T>(pow_a, pow_a * x.b * std::log(alpha));
}


/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(const Dual<T>& x, const Dual<T>& alpha) noexcept -> Dual<T> {
    T pow_a = std::pow(x.a, alpha.a);
    return Dual<T>(pow_a, pow_a * (alpha.b * std::log(x.a) + alpha.a * x.b / x.a));
}

/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(const HyperDual2<T>& x, T alpha) noexcept -> HyperDual2<T> {
    T pow_a = std::pow(x.a, alpha);
    return HyperDual2<T>(
        pow_a, 
        x.b * alpha * std::pow(x.a, alpha - 1.0),
        x.c * alpha * std::pow(x.a, alpha - 1.0),
        x.d * alpha * std::pow(x.a, alpha - 1.0) + x.b * x.c * alpha * (alpha - 1.0) * std::pow(x.a, alpha - 2.0)
    );
}

/// @brief  Power function for dual numbers.
template<std::floating_point T>
constexpr auto pow(T alpha, const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    T pow_a = std::pow(alpha, x.a);
    return HyperDual2<T>(
        pow_a, 
        pow_a * x.b * std::log(alpha), 
        pow_a * x.c * std::log(alpha),
        pow_a * x.d * std::log(alpha) + pow_a * x.b * x.c * std::log(alpha) * std::log(alpha)
    );
}


/// @brief  Square root function for dual numbers.
template<std::floating_point T>
auto sqrt(const Dual<T>& x) noexcept -> Dual<T> {
    T sqrt_a = std::sqrt(x.a);
    return Dual<T>(sqrt_a, x.b / (2.0 * sqrt_a));
}

/// @brief  Square root function for dual numbers.
template<std::floating_point T>
auto sqrt(const HyperDual2<T>& x) noexcept -> HyperDual2<T> {
    T sqrt_a = std::sqrt(x.a);
    return HyperDual2<T>(
        sqrt_a,
        x.b / (2.0 * sqrt_a),
        x.c / (2.0 * sqrt_a),
        x.d / (2.0 * sqrt_a) - x.b * x.c / (4.0 * x.a * sqrt_a)
    );
}

}

#endif