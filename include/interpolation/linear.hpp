#pragma once
#ifndef MATHLIB_INTERPOLATION_LINEAR_HPP
#define MATHLIB_INTERPOLATION_LINEAR_HPP

#include <array>
#include <vector>
#include <utility>
#include <concepts>
#include "../../third_party/lalib/include/vec.hpp"
#include "../../third_party/lalib/include/mat.hpp"
#include "../../third_party/lalib/include/solver/tri_diag.hpp"

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

	auto n_nodes() const noexcept -> size_t;
	auto n_segments() const noexcept -> size_t;

	/// @brief Returns the domain of this function.
	auto domain() const noexcept -> std::pair<double, double>;

private:
	std::vector<std::pair<double, T>> _nodes;
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
	for (auto i = 1u; i < this->_nodes.size(); ++i) {
		// Each data must be sorted in ascending order.
		assert(this->_nodes[i].first >= this->_nodes[i - 1].first && "dataset must be sorted in ascending order");

		if (x < this->_nodes[i].first) {
			auto&& s0 = this->_nodes[i - 1].first;
			auto&& s1 = this->_nodes[i].first;
			auto ds = s1 - s0;
			auto s_loc = (x - s0) / ds;

			auto tmp = s_loc * this->_nodes[i].second + (1 - s_loc) * this->_nodes[i - 1].second;
			return tmp;
		}
	}
	return this->_nodes.back().second;
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

template <typename T>
inline auto Linear<T>::domain() const noexcept -> std::pair<double, double>
{
	auto dom = std::make_pair(
		this->_nodes.front().first,
		this->_nodes.back().first
	);
	return dom;
}
}

#endif