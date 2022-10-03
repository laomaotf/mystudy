#pragma once
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
namespace logreg
{
	/////////////////////////////////////////////////////////////
	//W的维度和x一致，b是标量
	//每次取所有x的第n维数据组成一个向量，用这个向量计算W的第n维的值

	struct INPUT
	{
		std::vector< std::vector<float> > xs;
		std::vector< float> ys;
	};

	struct OUTPUT
	{
		std::vector<float> w;
		float b;
	};

	int Solve(INPUT& input, OUTPUT& output);

	std::vector<float> Predict(INPUT& input, std::vector<float> w, float b);
	float CalcNegLogCrossEntropyLoss(std::vector<float>& truths, std::vector<float>& preds);
	float CalAccuracy(std::vector<float>& truths, std::vector<float>& preds, float thresh = 0.5);
};
