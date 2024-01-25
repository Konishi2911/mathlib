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
	/// @param data 
	Linear(const std::vector<std::pair<double, T>>& data) noexcept;
	Linear(std::vector<std::pair<double, T>>&& data) noexcept;

	auto operator()(double x) const noexcept -> T;

	auto n_nodes() const noexcept -> size_t;
	auto n_segments() const noexcept -> size_t;

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
	for (auto i = 1u; i < this->_nodes.size(); ++i) {
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

}

#endif