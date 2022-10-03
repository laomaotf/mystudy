#include "logreg.h"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <numeric>

const cv::Size CANVAS_SIZE = cv::Size(512, 512);

const float data_range = 1;

float CalDistance(cv::Point2f p1, cv::Point2f p2)
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y,2));
}

int load_data(std::vector<std::vector<float>>& xs_train, std::vector<float>& ys_train,
			std::vector<std::vector<float>>& xs_test, std::vector<float>& ys_test)
{
	cv::Point2f center_a(CANVAS_SIZE.width / 4, CANVAS_SIZE.height / 4), center_b(CANVAS_SIZE.width * 3 / 4, CANVAS_SIZE.height * 3 / 4);

	center_a.x = (center_a.x - CANVAS_SIZE.width/2) * 1.0 / CANVAS_SIZE.width * data_range;
	center_b.x = (center_b.x - CANVAS_SIZE.width/2) * 1.0 / CANVAS_SIZE.width * data_range;

	center_a.y = (center_a.y - CANVAS_SIZE.height/2) * 1.0 / CANVAS_SIZE.height * data_range;
	center_b.y = (center_b.y - CANVAS_SIZE.height/2) * 1.0 / CANVAS_SIZE.height * data_range;



	int sample_num = 0;
	for (int y = 0; y < CANVAS_SIZE.height; y += CANVAS_SIZE.height / 10)
	{
		for (int x = 0; x < CANVAS_SIZE.width; x += CANVAS_SIZE.width / 10)
		{
			cv::Point2f pt((x - CANVAS_SIZE.width/2)*1.0/CANVAS_SIZE.width * data_range, (y - CANVAS_SIZE.height/2)*1.0/CANVAS_SIZE.height * data_range);
			float da = CalDistance(center_a, pt);
			float db = CalDistance(center_b, pt);
			int label = da < db ? 0 : 1;
			if (sample_num % 2 == 0)
			{
				xs_train.push_back(std::vector<float>{pt.x,pt.y});
				ys_train.push_back(label);
			}
			else
			{
				xs_test.push_back(std::vector<float>{ pt.x, pt.y});
				ys_test.push_back(label);
			}
			sample_num += 1;
		}
	}
	return 0;
}


float CalLoss(logreg::INPUT input, std::vector<float>& w, float b)
{
	std::vector<float> preds = logreg::Predict(input, w, b);
	return logreg::CalcNegLogCrossEntropyLoss(input.ys, preds);
}

float CalAcc(logreg::INPUT input, std::vector<float>& w, float b)
{
	std::vector<float> preds = logreg::Predict(input, w, b);
	float acc = logreg::CalAccuracy(input.ys, preds);
	return acc;
}

int main(int argc, char* argv[])
{
	logreg::INPUT input;
	logreg::OUTPUT output;
	logreg::INPUT test;


	load_data(input.xs, input.ys, test.xs, test.ys);

	/// /////////////////////////////////////////////////
	/// solver
	logreg::Solve(input, output);



	/////////////////////////////////////////////////
	//validate
	float loss_train = CalLoss(input,output.w, output.b);
	float loss_test = CalLoss(test, output.w, output.b);

	std::cout << "=============================================" << std::endl;
	std::cout << "train CELoss: " << loss_train << std::endl << "test CELoss: " << loss_test << std::endl;


	float acc_train = CalAcc(input, output.w, output.b);
	float acc_test = CalAcc(test, output.w, output.b);

	std::cout << "=============================================" << std::endl;
	std::cout << "train acc: " << acc_train * 100 << "%" << std::endl << "test acc: " << acc_test * 100 <<"%" << std::endl;

	return 0;
} 