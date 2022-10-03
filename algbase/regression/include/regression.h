#pragma once
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
namespace regression
{
	/////////////////////////////////////////////////////////////
	//W的维度和x一致，b是标量
	//每次取所有x的第n维数据组成一个向量，用这个向量计算W的第n维的值

	struct INPUT
	{
		std::vector< std::vector<float> > xs; //自变量X
		std::vector< float> ys; //因变量Y
	};

	struct OUTPUT
	{
		std::vector<float> w; //len(w) == len(xs[0])
		float b;
	};

	int LeastSquareMean(INPUT& input, OUTPUT& output);

	int GradientDecent(INPUT& input, OUTPUT& output);
};
