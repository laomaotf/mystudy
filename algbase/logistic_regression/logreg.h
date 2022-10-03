#pragma once
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
namespace logreg
{
	/////////////////////////////////////////////////////////////
	//W��ά�Ⱥ�xһ�£�b�Ǳ���
	//ÿ��ȡ����x�ĵ�nά�������һ���������������������W�ĵ�nά��ֵ

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
