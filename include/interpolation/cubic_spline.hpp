#pragma once
#ifndef MATHLIB_INTERPOLATION_CUBIC_SPLINE_HPP
#define MATHLIB_INTERPOLATION_CUBIC_SPLINE_HPP

#include <array>
#include <vector>
#include <utility>
#include <concepts>
#include "../../third_party/lalib/include/vec.hpp"
#include "../../third_party/lalib/include/mat.hpp"
#include "../../third_party/lalib/include/solver/tri_diag.hpp"

namespace mathlib {

template<typename T>
struct CubicSpline {
public:
	/// @brief 	Constructs an instance of cibic spline with given data.
	/// @param data 
	CubicSpline(const std::vector<std::pair<double, T>>& data) noexcept;
	CubicSpline(std::vector<std::pair<double, T>>&& data) noexcept;

	auto operator()(double x) const noexcept -> T;

	auto n_nodes() const noexcept -> size_t;
	auto n_segments() const noexcept -> size_t;

	/// @brief Returns the domain range
	auto domain() const noexcept -> std::pair<double, double>;

private:
	std::vector<std::array<T, 4>> _c;
	std::vector<std::pair<double, T>> _nodes;

	static auto __calc_coefficients(const std::vector<std::pair<double, T>>& data) noexcept -> std::vector<std::array<T, 4>>;
	auto __get_segment_id(double x) const noexcept -> size_t;
};


template <typename T>
inline CubicSpline<T>::CubicSpline(const std::vector<std::pair<double, T>> &data) noexcept:
	_c(__calc_coefficients(data)), _nodes(data)
{ }

template <typename T>
inline CubicSpline<T>::CubicSpline(std::vector<std::pair<double, T>> &&data) noexcept:
	_c(__calc_coefficients(data)), _nodes(std::move(data))
{ }

template <typename T>
inline auto CubicSpline<T>::operator()(double x) const noexcept -> T
{
	auto sid = this->__get_segment_id(x);
	auto x0 = this->_nodes[sid].first;
	auto f = this->_c[sid][0] + this->_c[sid][1] * (x - x0) + this->_c[sid][2] * std::pow(x - x0, 2) + this->_c[sid][3] * std::pow(x - x0, 3);
	return f;
}

template <typename T>
inline auto CubicSpline<T>::n_nodes() const noexcept -> size_t
{
	auto n_nodes = this->_nodes.size();
	return n_nodes;
}

template <typename T>
inline auto CubicSpline<T>::n_segments() const noexcept -> size_t
{
	auto n_segs = this->n_nodes() - 1;
	return n_segs;
}

template <typename T>
inline auto CubicSpline<T>::domain() const noexcept -> std::pair<double, double>
{
	auto domain = std::make_pair(this->_nodes.front().first, this->_nodes.back().first);
	return domain;
}

template <typename T>
inline auto CubicSpline<T>::__calc_coefficients(const std::vector<std::pair<double, T>> &data) noexcept -> std::vector<std::array<T, 4>>
{
	auto n_seg = data.size() - 1;

	// create RHS vector
	auto rhs_tmp = std::vector<T>();
	rhs_tmp.reserve(n_seg - 1);
	for (auto i = 0u; i < n_seg - 1; ++i) {
		auto dx1 = data[i + 2].first - data[i + 1].first;
		auto dx0 = data[i + 1].first - data[i].first;
		auto df1 = data[i + 2].second - data[i + 1].second;
		auto df0 = data[i + 1].second - data[i].second;

		rhs_tmp.emplace_back(df1 / dx1 - df0 / dx0);
	}
	auto rhs = lalib::DynVec<T>(std::move(rhs_tmp));

	// create coefficient matrix
	auto cm = std::vector<T>();
	auto cl = std::vector<T>();
	auto cu = std::vector<T>();
	cm.reserve(n_seg - 1);
	cl.reserve(n_seg - 2);
	cu.reserve(n_seg - 2);
	for (auto i = 0u; i < n_seg - 1; ++i) {
		auto dx1 = data[i + 2].first - data[i + 1].first;
		auto dx0 = data[i + 1].first - data[i].first;
		cm.emplace_back((dx1 + dx0) * 2.0 / 3.0);
	}
	for (auto i = 0u; i < n_seg - 2; ++i) {
		auto dx1 = data[i + 2].first - data[i + 1].first;
		cu.emplace_back(dx1 / 3.0);
	}
	for (auto i = 1u; i < n_seg - 1; ++i) {
		auto dx0 = data[i + 1].first - data[i].first;
		cl.emplace_back(dx0 / 3.0);
	}
	auto mat = lalib::DynTriDiagMat(std::move(cl), std::move(cm), std::move(cu));

	// Solve linear equation
	auto solver = lalib::solver::TriDiag(std::move(mat));
	solver.solve_linear(rhs, rhs);

	// Create coefficient set
	auto coeffs = std::vector<std::array<T, 4>>();
	coeffs.reserve(n_seg);
	{
		auto dx = (data[1].first - data[0].first);
		coeffs.emplace_back(
			std::array {
				data[0].second,
				(data[1].second - data[0].second) / dx - rhs[0] * dx / 3.0,
				lalib::Zero<T>::value(),
				rhs[0] / (3.0 * dx)	
			}
		);
	}
	for (auto i = 1u; i < n_seg - 1; ++i) {
		auto dx = data[i + 1].first - data[i].first;
		auto df = data[i + 1].second - data[i].second;
		coeffs.emplace_back(
			std::array {
				data[i].second,
				df / dx - (rhs[i] + 2.0 * rhs[i - 1]) * dx / 3.0,
				rhs[i - 1],
				(rhs[i] - rhs[i - 1]) / (3.0 * dx)
			}
		);
	}
	{
		auto dx = data[n_seg].first - data[n_seg - 1].first;
		auto df = data[n_seg].second - data[n_seg - 1].second;
		coeffs.emplace_back(
			std::array {
				data[n_seg - 1].second,
				df / dx - 2.0 * rhs[n_seg - 2] * dx / 3.0,
				rhs[n_seg - 2],
				-rhs[n_seg - 2] / (3.0 * dx)	
			}
		);
	}

	return coeffs;
}

template <typename T>
inline auto CubicSpline<T>::__get_segment_id(double x) const noexcept -> size_t
{
	auto n_nodes = this->_nodes.size();
	for (auto i = 1u; i < n_nodes; ++i) {
		if (x < this->_nodes[i].first) {
			return i - 1;
		}
	}
	return n_nodes - 2;	// last segment
}

}

#endif
