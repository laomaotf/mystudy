/*
reference to "Optimizing Connected Component Labeling Algorithms" by Keshen Wu, Ekow Otoo and Arie ShoShani
*/
#pragma once
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"

#include "boost/log/trivial.hpp"
#include "boost/log/utility/setup/file.hpp"
#include <boost/log/utility/setup/common_attributes.hpp>
namespace logging = boost::log;
namespace keywords = boost::log::keywords;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
namespace beam_decoder
{
	struct INPUT
	{
		cv::Mat probs; //���ʾ�����������Time����������Symbol
		int beam_size; //beam width
	};
	
	struct OUTPUT
	{
		float prob; //����
		std::vector<int> beam; //��������beam
	};

	int FirstPassLM(INPUT&  input, OUTPUT& output, int blank = 0);
};
