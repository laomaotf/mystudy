#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include "regression.h"
std::vector<float> get_dim(int dim, regression::INPUT& input)
{
	std::vector<float> one_dim;
	for (auto& one : input.xs)
	{
		one_dim.push_back(one[dim]);
	}
	return one_dim;
}
