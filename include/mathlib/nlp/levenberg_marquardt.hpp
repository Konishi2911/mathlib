#pragma once 
#ifndef MATHLIB_LEVENBERG_MARQUARDT_HPP
#define MATHLIB_LEVENBERG_MARQUARDT_HPP

#include "mathlib/nlp/cost_function.hpp"
#include "mathlib/nlp/solver_common.hpp"
#include "lalib/solver/cholesky_factorization.hpp"
#include "lalib/mat.hpp"
#include <limits>

namespace mathlib::nlp {

struct LevenbregMarquardt {
    LevenbregMarquardt() noexcept;

    template<typename T, CostFunc<T> F>
    auto solve(T init, F&& func, size_t max_iter, double tol) const -> NlpResult<T>;
};

namespace _lm_ {
    template<typename T> struct _LMSubSolver_ { };

    template<std::floating_point T> 
    struct _LMSubSolver_<T> {
        static auto solve(double lambda, const T&, const T& grad, const T& hess) noexcept -> T {
            auto s = -grad / (hess + lambda);
            return s;
        }
    };

    template<std::floating_point T> 
    struct _LMSubSolver_<lalib::DynVec<T>> {
        static auto solve(double lambda, const lalib::DynVec<T>&, const lalib::DynVec<T>& grad, const lalib::DynMat<T>& hess) noexcept -> lalib::DynVec<T> {
            auto l = lalib::DynMatD::diag(lambda, hess.shape().first);
            auto cholesky = lalib::solver::DynModCholeskyFactorization(hess + l);
            auto s = cholesky.solve_linear(-grad);
            return s;
        }
    };
}

inline LevenbregMarquardt::LevenbregMarquardt() noexcept {}

template<typename T, CostFunc<T> F>
inline auto LevenbregMarquardt::solve(T init, F&& func, size_t max_iter, double tol) const -> NlpResult<T> {
    	const auto nu = 2.0;
	auto x = init;
	auto lambda = 1.0;

    auto prev_cost = std::numeric_limits<double>::infinity();
	auto cost = func(x);
	
	for (auto k = 0l; k < static_cast<int32_t>(max_iter); ++k) {
		const auto grad = func.grad(x);
		const auto hessian = func.hessian(x);
		const auto s_fast = _lm_::_LMSubSolver_<T>::solve(lambda / nu, x, grad, hessian);
		const auto s_slow = _lm_::_LMSubSolver_<T>::solve(lambda, x, grad, hessian);
		const auto x_fast = x + s_fast;
		const auto x_slow = x + s_slow;
		const auto cost_fast = func(x_fast);
		const auto cost_slow = func(x_slow);

		// Update trust region
		if (cost_slow < cost) {
            prev_cost = cost;
			if (cost_fast < cost_slow) {
				lambda /= nu;
				x = x_fast;
				cost = cost_fast;
			} else {
				x = x_slow;
				cost = cost_slow;
			}
            
            // Convergence check
            auto resi = std::abs(cost - prev_cost);
            if (resi < tol) { 
                auto result = NlpResult<T>(true, k, x, cost, resi); 
                return result;
            }
		} else {
			lambda *= nu;
            --k;
		}
	}
    auto result = NlpResult<T>(false, max_iter, x, cost, std::abs(cost - prev_cost));
    return result;
}

}

#endif 