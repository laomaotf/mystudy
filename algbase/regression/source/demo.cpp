#include "regression.h"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <numeric>

#define W (3.43)
#define B (1.3)

int load_data(std::vector<std::vector<float>>& xs_train, std::vector<float>& ys_train,
			std::vector<std::vector<float>>& xs_test, std::vector<float>& ys_test, bool use_linear_model=true)
{
	const int sample_num = 100;

	for (int sample = 0; sample < sample_num; sample++)
	{
		float x = sample / float(sample_num);
		float y;
		if(use_linear_model)
			y = W * x + B;
		else
			y = sin( float(x) * 2 /sample_num  * 3.14);
		if (sample % 10 == 0)
		{
			xs_train.push_back(std::vector<float>{x});
			ys_train.push_back(y);
		}
		else
		{
			xs_test.push_back(std::vector<float>{x});
			ys_test.push_back(y);

		}
	}
	return 0;
}

std::vector<float> predict(regression::OUTPUT& output, std::vector<std::vector<float>>& xs)
{
	std::vector<float> ypreds(xs.size(), 0);
	for (int sample = 0; sample < xs.size(); sample++)
	{
		float sum_all = output.b;
		for (int k = 0; k < output.w.size(); k++)
		{
			sum_all += output.w[k] * xs[sample][k];
		}
		ypreds[sample] = sum_all;
	}
	return ypreds;
}

float calc_mse(regression::OUTPUT& output, regression::INPUT& test)
{
	/// ///////////////////////////////////////////////////////////////////
	//prediction
	std::vector<float> ypreds = predict(output, test.xs);

	/////////////////////////////////////////////////////////////////////////////
	//º∆À„MSE
	//1: (gt - x)
	std::vector<float> neg_ypreds;
	std::transform(ypreds.begin(), ypreds.end(), std::back_inserter(neg_ypreds), [](float val) { return -val;});
	std::vector<float> subs;
	std::transform(neg_ypreds.begin(), neg_ypreds.end(), test.ys.begin(), std::back_inserter(subs), std::plus<float>());

	//2: (gt - x)^2
	std::vector<float> subs2;
	std::transform(subs.begin(), subs.end(), std::back_inserter(subs2), [](float val) { return val * val;});

	float mse = std::accumulate(subs2.begin(), subs2.end(), 0.0) / subs2.size();
	return mse;
}

int main(int argc, char* argv[])
{
	regression::INPUT input;
	regression::OUTPUT output;
	regression::INPUT test;

	bool use_linear_model = true;
	bool use_gradient_decent = true;

	std::cout << "LSM:" << (use_linear_model? "ON":"OFF") << std::endl;
	std::cout << "SGD:" << (use_gradient_decent? "ON":"OFF") << std::endl;

	load_data(input.xs, input.ys, test.xs, test.ys, use_linear_model);

	/// /////////////////////////////////////////////////
	/// solver
	if (use_gradient_decent)
		regression::GradientDecent(input, output);
	else
		regression::LeastSquareMean(input, output);

	if (use_linear_model)
	{
		std::cout << "=============================================" << std::endl;
		std::cout << "\t gt: w = " << W << ", b = " << B << std::endl;
		std::cout << "\t pred: w = " << output.w[0] << ", b = " << output.b << std::endl;
	}


	/////////////////////////////////////////////////
	//validate
	float mse_train = calc_mse(output, input);
	float mse_test = calc_mse(output, test);

	std::cout << "=============================================" << std::endl;
	std::cout << "train MSE: " << mse_train << std::endl << "test MSE: " << mse_test << std::endl;

	return 0;
} 