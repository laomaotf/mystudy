#pragma once
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
namespace regression
{
	/////////////////////////////////////////////////////////////
	//W��ά�Ⱥ�xһ�£�b�Ǳ���
	//ÿ��ȡ����x�ĵ�nά�������һ���������������������W�ĵ�nά��ֵ

	struct INPUT
	{
		std::vector< std::vector<float> > xs; //�Ա���X
		std::vector< float> ys; //�����Y
	};

	struct OUTPUT
	{
		std::vector<float> w; //len(w) == len(xs[0])
		float b;
	};

	int LeastSquareMean(INPUT& input, OUTPUT& output);

	int GradientDecent(INPUT& input, OUTPUT& output);
};
