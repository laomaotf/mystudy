/*
* 最小二乘法
*/

#include "vector_ops.h"


namespace regression
{
	int LeastSquareMean(INPUT& input, OUTPUT& output)
	{
		output.w.resize(input.xs[0].size());
		for (int dim = 0; dim < input.xs[0].size(); dim++)
		{
			std::vector<float> one = get_dim(dim, input);
			float m = one.size();

			VEC_OP<float> one_op(one);
			float xbar = one_op.mean();
			std::vector<float> x_sub_xbar = one_op - xbar;
			std::vector<float> x2 = one_op.pow2();
			float xsum = one_op.sum();


			VEC_OP<float> x2_op(x2);
			float x2sum = x2_op.sum();

			float sum_y_mul_x_sub_xbar = std::inner_product(input.ys.begin(), input.ys.end(), x_sub_xbar.begin(), 0.0f);

			float w = sum_y_mul_x_sub_xbar / (x2sum - xsum * xsum / m);
			output.w[dim] = w;
		}

		///////////////////////////////////////////////////////
		//计算b
		{
			output.b = 0;
			float m = input.xs.size();
			for (int sample = 0; sample < input.xs.size(); sample++)
			{
				std::vector<float>& one_sample = input.xs[sample];

				float wx = std::inner_product(one_sample.begin(), one_sample.end(), output.w.begin(), 0.0f);

				float y_sub_wx = input.ys[sample] - wx;

				output.b += y_sub_wx;
			}
			output.b /= m;
		}

		return 0;
	}

}
