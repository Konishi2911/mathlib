#pragma once
#ifndef MATHLIB_INTERPOLATION_LINEAR_HPP
#define MATHLIB_INTERPOLATION_LINEAR_HPP

#include <array>
#include <vector>
#include <utility>
#include <concepts>
#include "lalib/vec.hpp"
#include "lalib/mat.hpp"
#include "lalib/solver/tri_diag.hpp"
#include "mathlib/dual_number.hpp"

namespace mathlib::intrpl {

template<typename T>
struct Linear {
public:
	/// @brief 	Constructs an instance of cibic spline with given data.
	/// @param data a dataset x, y pairs as elements.
	Linear(const std::vector<std::pair<double, T>>& data) noexcept;
	Linear(std::vector<std::pair<double, T>>&& data) noexcept;

	/// @brief Returns the interpolated value at given 'x'.
	auto operator()(double x) const noexcept -> T;
	auto operator()(const Dual<double>& x) const noexcept -> Dual<T>;
	auto operator()(const HyperDual2<double>& x) const noexcept -> HyperDual2<T>;

	auto n_nodes() const noexcept -> size_t;
	auto n_segments() const noexcept -> size_t;

	/// @brief Returns the nodes of this function.
	auto nodes() const noexcept -> const std::vector<std::pair<double, T>>&;

	/// @brief Returns the domain of this function.
	auto domain() const noexcept -> std::pair<double, double>;

private:
	std::vector<std::pair<double, T>> _nodes;

	auto _find_segment(double x) const noexcept -> size_t;
};

template <typename T>
inline Linear<T>::Linear(const std::vector<std::pair<double, T>> &data) noexcept:
    _nodes(data)
{
}

template <typename T>
inline Linear<T>::Linear(std::vector<std::pair<double, T>> &&data) noexcept:
    _nodes(std::move(data))
{
}
template <typename T>
inline auto Linear<T>::operator()(double x) const noexcept -> T
{
	if (x < this->_nodes.front().first) {
		return this->_nodes.front().second;
	}

	size_t i = this->_find_segment(x);

	if (i == this->_nodes.size() - 1) {
		return this->_nodes.back().second;
	}

	auto&& s0 = this->_nodes[i].first;
	auto&& s1 = this->_nodes[i + 1].first;
	auto&& f0 = this->_nodes[i].second;
	auto&& f1 = this->_nodes[i + 1].second;
	auto ds = s1 - s0;
	auto s_loc = (x - s0) / ds;

	auto tmp = s_loc * f1 + (1 - s_loc) * f0;
	return tmp;
}

template<typename T>
inline auto Linear<T>::operator()(const Dual<double>& x) const noexcept -> Dual<T> {
	if (x.a < this->_nodes.front().first) {
		T f = this->_nodes.front().second;

		auto&& s0 = this->_nodes[0].first;
		auto&& s1 = this->_nodes[1].first;
		auto&& f0 = this->_nodes[0].second;
		auto&& f1 = this->_nodes[1].second;
		auto ds = s1 - s0;
		auto df = f1 - f0;

		return Dual<T>(f, x.b * df / ds);
	}

	size_t i = this->_find_segment(x.a);

	if (i == this->_nodes.size() - 1) {
		T f = this->_nodes.back().second;

		auto&& s0 = this->_nodes[i - 1].first;
		auto&& s1 = this->_nodes[i].first;
		auto&& f0 = this->_nodes[i - 1].second;
		auto&& f1 = this->_nodes[i].second;
		auto ds = s1 - s0;
		auto df = f1 - f0;

		return Dual<T>(f, x.b * df / ds);
	}

	auto&& s0 = this->_nodes[i].first;
	auto&& s1 = this->_nodes[i + 1].first;
	auto&& f0 = this->_nodes[i].second;
	auto&& f1 = this->_nodes[i + 1].second;
	auto ds = s1 - s0;
	auto df = f1 - f0;
	auto s_loc = (x.a - s0) / ds;

	auto f = s_loc * f1 + (1 - s_loc) * f0;
	return Dual<T>(f, x.b * df / ds);
}

template<typename T>
inline auto Linear<T>::operator()(const HyperDual2<double>& x) const noexcept -> HyperDual2<T> {
	if (x.a < this->_nodes.front().first) {
		T f = this->_nodes.front().second;

		auto&& s0 = this->_nodes[0].first;
		auto&& s1 = this->_nodes[1].first;
		auto&& f0 = this->_nodes[0].second;
		auto&& f1 = this->_nodes[1].second;
		auto ds = s1 - s0;
		auto df = f1 - f0;

		return HyperDual2<T>(f, x.b * df / ds, x.c * df / ds, x.d * df / ds);
	}

	size_t i = this->_find_segment(x.a);

	if (i == this->_nodes.size() - 1) {
		T f = this->_nodes.back().second;

		auto&& s0 = this->_nodes[i - 1].first;
		auto&& s1 = this->_nodes[i].first;
		auto&& f0 = this->_nodes[i - 1].second;
		auto&& f1 = this->_nodes[i].second;
		auto ds = s1 - s0;
		auto df = f1 - f0;

		return HyperDual2<T>(f, x.b * df / ds, x.c * df / ds, x.d * df / ds);
	}

	auto&& s0 = this->_nodes[i].first;
	auto&& s1 = this->_nodes[i + 1].first;
	auto&& f0 = this->_nodes[i].second;
	auto&& f1 = this->_nodes[i + 1].second;
	auto ds = s1 - s0;
	auto df = f1 - f0;
	auto s_loc = (x.a - s0) / ds;

	auto f = s_loc * f1 + (1 - s_loc) * f0;
	return HyperDual2<T>(f, x.b * df / ds, x.c * df / ds, x.d * df / ds);
}

template <typename T>
inline auto Linear<T>::n_nodes() const noexcept -> size_t
{
    return this->_nodes.size();
}
template <typename T>
inline auto Linear<T>::n_segments() const noexcept -> size_t
{
    return this->_nodes.size() - 1;
}

template<typename T>
inline auto Linear<T>::nodes() const noexcept -> const std::vector<std::pair<double, T>>&
{
	return this->_nodes;
}

template <typename T>
inline auto Linear<T>::domain() const noexcept -> std::pair<double, double>
{
	auto dom = std::make_pair(
		this->_nodes.front().first,
		this->_nodes.back().first
	);
	return dom;
}

template <typename T>
inline auto Linear<T>::_find_segment(double x) const noexcept -> size_t
{
	for (auto i = 1u; i < this->_nodes.size(); ++i) {
		// Each data must be sorted in ascending order.
		assert(this->_nodes[i].first >= this->_nodes[i - 1].first && "dataset must be sorted in ascending order");

		if (x < this->_nodes[i].first) {
			return i - 1;
		}
	}
	return this->_nodes.size() - 1;
}

}

#endif