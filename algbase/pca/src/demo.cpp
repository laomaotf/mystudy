#include "pca.h"
#include <cmath>
#include <string>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <vector>
#include "opencv2/opencv.hpp"


#define INPUT_SIZE (14)

int load_data(std::string data_dir, std::vector< std::vector<float> >& train_data, std::vector<int>& train_label,
	std::vector< std::vector<float> >& test_data, std::vector<int>& test_label)
{

	int MAX_N_EACH_CLASS = 500;
	std::vector<int> labels = { 1,2,3,4,5,6,7,8,9,0 };
	

	const int resized_size = INPUT_SIZE;

	for (auto label : labels)
	{
		std::ostringstream pattern;
		pattern << data_dir <<"\\"<<label<< "\\*.jpg";

		std::vector<cv::String> paths;
		cv::glob(pattern.str().c_str(), paths, false);

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
				train_data.push_back(data);
				train_label.push_back(label);
			}
			else
			{
				test_data.push_back(data);
				test_label.push_back(label);
			}
			num++;
		}
	}

	return 0;
}

void visualize_eigvecs(std::shared_ptr<pca::MODEL> model)
{
	int num = model->eigvecs.size();
	int numx = 10, numy = (num+numx-1) / numx;
	cv::Mat vis = cv::Mat::zeros(cv::Size(numx * INPUT_SIZE, numy * INPUT_SIZE), CV_8UC1) + 255;
	for (int y = 0; y < numy; y ++)
	{
		for (int x = 0; x < numx; x++)
		{
			if (y * numx + x >= num) break;

			auto& eigvec = model->eigvecs[y * numx + x];
			int x0 = x * INPUT_SIZE, y0 = y * INPUT_SIZE;
			for (int dy = 0; dy < INPUT_SIZE; dy++)
			{
				for (int dx = 0; dx < INPUT_SIZE; dx++)
				{
					int pos = dy * INPUT_SIZE + dx;
					int val = eigvec[pos] + model->mean[pos];
					val = std::min<int>(std::max<int>(val, 0), 255);
					vis.at<uchar>(dy + y0, dx + x0) = val;
				}
			}
		}
	}
	cv::resize(vis, vis, cv::Size(512, 512), 0, 0, cv::INTER_AREA);
	cv::imshow("pca-eigvecs", vis);
}

void visualize2D(std::vector< std::vector<float> >& data, std::vector<int>& labels)
{
	const int canvas_size = 512;
	if (data[0].size() != 2)
		throw std::runtime_error("must be 2d for visual");
	int num = data.size();

	float min_val[2] = { FLT_MAX, FLT_MAX}, max_val[2] = { -FLT_MAX, -FLT_MAX };
	for (int y = 0; y < num; y++)
	{
		for (int k = 0; k < 2; k++)
		{
			min_val[k] = std::min<float>(min_val[k], data[y][k]);
			max_val[k] = std::max<float>(max_val[k], data[y][k]);
		}
	}

	int label_max = 0;
	for (auto& label : labels)
	{
		label_max = std::max<int>(label, label_max);
	}
	srand(label_max + 1);
	std::vector< cv::Scalar > colors;
	for (int k = 0; k <= label_max; k++)
	{
		colors.push_back( 
			CV_RGB(rand() % 100 + 150, rand() % 255, rand() % 255)
		);
	}

	cv::Mat canvas = cv::Mat::zeros(cv::Size(canvas_size, canvas_size), CV_8UC3) + CV_RGB(255,255,255);


	/*
	* 对输入数据缩放后可视化
	*/
	for(int k = 0; k < data.size(); k++)
	{
		auto& xy = data[k];
		int label = labels[k];
		int x = (xy[0] - min_val[0]) * canvas_size * 0.8 / (max_val[0] - min_val[0]) + canvas_size * 0.1;
		int y = (xy[1] - min_val[1]) * canvas_size * 0.8 / (max_val[1] - min_val[1]) + canvas_size * 0.1;

		cv::Scalar color = colors[label];

		cv::circle(canvas, cv::Point(x, y), 3, color, -1);

	}

	cv::imshow("pca", canvas);

	return;

}

int main(int argc, char* argv[])
{
	std::vector< std::vector<float> > train_data, test_data;
	std::vector< int > train_label, test_label;
	std::string data_root = getenv("DATASET_ROOT_DIR");
	data_root += "\mnist\\images";
	load_data(data_root, train_data, train_label, test_data, test_label);

	std::shared_ptr<pca::MODEL> model = pca::BuildModel(train_data);

	std::vector< std::vector<float> > projs;
	for (int k = 0; k < test_data.size(); k++)
	{
		projs.push_back(
			pca::Project(model, test_data[k], 2)
		);
	}

	visualize_eigvecs(model);
	visualize2D(projs, test_label);

	cv::waitKey(-1);
	return 0;
} 