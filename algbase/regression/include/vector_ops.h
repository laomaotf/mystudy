#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include "regression.h"

std::vector<float> get_dim(int dim, regression::INPUT& input);


template<class T>
class VEC_OP
{
	std::vector<T> _data;
public:
	VEC_OP(std::vector<T>& data)
	{
		std::copy(data.begin(), data.end(), std::back_inserter(_data));
	}
	~VEC_OP()
	{
		_data.clear();
	}
public:
	float sum()
	{
		float val = std::accumulate(_data.begin(), _data.end(), 0.0f);
		return val;
	}
	float mean()
	{
		float val = std::accumulate(_data.begin(), _data.end(), 0.0f) / _data.size();
		return val;
	}
	std::vector<T> pow2()
	{
		std::vector<T> output;
		std::transform(_data.begin(), _data.end(), std::back_inserter(output), [](float val) {return val * val; });
		return output;

	}
	std::vector<T> operator-(T s)
	{
		std::vector<T> output;
		std::transform(_data.begin(), _data.end(), std::back_inserter(output), [s](float val) {return val - s; });
		return output;
	}

	std::vector<T> operator-(std::vector<T>& s)
	{
		std::vector<T> output(_data.size(), 0);
		for (int k = 0; k < _data.size(); k++)
		{
			output[k] = _data[k] - s[k];
		}
		return output;
	}

	std::vector<T> operator*(std::vector<T>& s)
	{
		std::vector<T> output(_data.size(), 0);
		for (int k = 0; k < _data.size(); k++)
		{
			output[k] = s[k] * _data[k];
		}
		return output;
	}


public:
	std::vector<T> data()
	{
		return _data;
	}
};