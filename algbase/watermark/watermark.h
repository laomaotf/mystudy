#pragma once
#include <vector>
#include "opencv2/opencv.hpp"

namespace watermark
{

	/*
	* Ƶ��ˮӡ��˼·��ֱ�Ӱ�ͼƬset��Ƶ���У�������watermarkͼƬ����ֵ�޸�image��Ƶ����Ϣ��
	* watermarkͼ�޸�Ƶ����Ϣʱ����������룬��������Ϣ�ֲ�������Ƶ�ʣ����������ˮӡ���ȶ���
	*/
	struct INPUT
	{
		cv::Mat image;
		cv::Mat watermark;
		float alpha;
	};

	struct OUTPUT
	{
		cv::Mat image;
		cv::Mat watermak;
	};
	int Encode(INPUT& input, OUTPUT& output);
};
