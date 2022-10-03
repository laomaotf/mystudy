#include "lda.h"
#include <cmath>
#include <string>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <vector>
#include "opencv2/opencv.hpp"


#define INPUT_SIZE (14)

int load_data(std::string data_dir, std::vector<std::vector< std::vector<float> >>& train_data, 
	std::vector<std::vector< std::vector<float> >>& test_data)
{


	int MAX_N_EACH_CLASS = 512;
	std::vector<int> labels = { 0,8 };
	

	const int resized_size = INPUT_SIZE;

	for (auto label : labels)
	{
		std::ostringstream pattern;
		pattern << data_dir <<"\\"<<label<< "\\*.jpg";

		std::vector<cv::String> paths;
		cv::glob(pattern.str().c_str(), paths, false);

		std::vector<std::vector<float>> test_data_cls, train_data_cls;
		int num = 0;
		for (auto& path : paths)
		{
			if (num >= MAX_N_EACH_CLASS) break;
			cv::Mat image = cv::imread(path.c_str(), 0);
			cv::resize(image, image, cv::Size(resized_size, resized_size), 0, 0, cv::INTER_CUBIC);
			
			std::vector<float> data;
			for(int y = 0; y < image.rows; y++)
			{
				for(int x = 0; x < image.cols; x++)
					data.push_back(image.at<uchar>(y,x));
			}

			if ((num % 2) == 0)
			{
				train_data_cls.push_back(data);
			}
			else
			{
				test_data_cls.push_back(data);
			}
			num++;
		}

		train_data.push_back(train_data_cls);
		test_data.push_back(test_data_cls);
	}

	return 0;
}

void visualize2D(std::vector< std::vector<float> >& data)
{
	const int canvas_size = 512;
	if (data.size() != 2)
		throw std::runtime_error("2 classes required");
	int num_cls = data.size();

	float min_val = FLT_MAX, max_val = FLT_MIN;
	for (int cls = 0; cls < num_cls; cls++)
	{
		auto p = std::minmax_element(data[cls].begin(), data[cls].end());
		min_val = std::min<float>(min_val, *p.first);
		max_val = std::max<float>(max_val, *p.second);
	}

	srand(num_cls);
	std::vector< cv::Scalar > colors;
	for (int k = 0; k < num_cls; k++)
	{
		colors.push_back( 
			CV_RGB(rand() % 100 + 150, rand() % 255, rand() % 255)
		);
	}

	cv::Mat canvas = cv::Mat::zeros(cv::Size(canvas_size, canvas_size), CV_8UC3) + CV_RGB(255,255,255);


	/*
	* 对输入数据缩放后可视化
	*/
	int rec = 0;
	int total = 0;
	for(int cls = 0; cls < num_cls; cls++)
	{
		total += data[cls].size();
		cv::Scalar color = colors[cls];
		for (int k = 0; k < data[cls].size(); k++)
		{
			float val = data[cls][k];
			rec += (cls == 0 && val > 0) || (cls == 1 && val < 0);
			int x = (val - min_val) * canvas_size * 0.8 / (max_val - min_val) + canvas_size * 0.1;
			int y = canvas_size / 2;

			cv::circle(canvas, cv::Point(x, y), 3, color, -1);
		}
	}

	std::ostringstream oss;
	oss << "lda rec = " << (int)(rec * 100 / total);
	cv::imshow(oss.str(), canvas);

	return;

}

int main(int argc, char* argv[])
{
	std::vector<std::vector< std::vector<float> >> train_data, test_data;
	std::string data_dir = getenv("DATASET_ROOT_DIR");
	data_dir += "\\mnist\\images";
	load_data(data_dir, train_data,test_data);



	std::shared_ptr<lda::MODEL> model = lda::BuildModel(train_data);

	std::vector<std::vector<float>> projs;
	for (int cls = 0; cls < test_data.size(); cls++)
	{
		std::vector<float> projs_cls;
		for(int k = 0; k < test_data[cls].size(); k++)
		{
			projs_cls.push_back(
				lda::Project(model, test_data[cls][k])
			);
		}
		projs.push_back(projs_cls);
	}

	visualize2D(projs);

	cv::waitKey(-1);
	return 0;
} 