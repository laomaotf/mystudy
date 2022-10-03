/*
* 梯度下降方法求解
*/
#include "vector_ops.h"

namespace regression
{
	std::vector<float> Predict(INPUT& input, std::vector<float> w, float b)
	{
		std::vector<float> preds(input.xs.size(), 0);
		for (int sample = 0; sample < input.xs.size(); sample++)
		{
			std::vector<float>& x = input.xs[sample];
			float y = std::inner_product(x.begin(), x.end(), w.begin(), b);
			preds[sample] = y;
		}
		return preds;
	}

	float CalMSELoss(std::vector<float>& truths, std::vector<float>& preds)
	{
		VEC_OP<float> truths_op(truths);
		std::vector<float> diff = truths_op - preds;
		std::vector<float> diff2;
		std::transform(diff.begin(), diff.end(), std::back_inserter(diff2), [](float val) {return val * val; });
		float loss = std::accumulate(diff2.begin(), diff2.end(), 0.0f);
		return loss / truths.size();
	}

	std::vector<float> CalGradientW(INPUT& input, std::vector<float>& last_w, float last_b)
	{
		std::vector<float> gradient(input.xs[0].size());
		int m = input.xs[0].size();
		for (int dim = 0; dim < input.xs[0].size(); dim++)
		{
			std::vector<float> one = get_dim(dim, input);

			VEC_OP<float> one_op(one);
			std::vector<float> x2 = one_op.pow2();
			float sum_x2 = std::accumulate(x2.begin(), x2.end(), 0.0f);

			VEC_OP<float> y_op(input.ys);
			std::vector<float> y_sub_b = y_op - last_b;
			float sum_y_sub_b_mul_x = std::inner_product(
				y_sub_b.begin(), y_sub_b.end(), one.begin(), 0.0f
			);

			gradient[dim] = (last_w[dim] * sum_x2 - sum_y_sub_b_mul_x) * 2 / input.xs.size();
		}
		return gradient;
	}

	float CalGradientB(INPUT& input, std::vector<float>& last_w, float last_b)
	{
		float sum_y_wx = 0;
		for (int sample = 0; sample < input.xs.size(); sample++)
		{
			std::vector<float>& x = input.xs[sample];
			float wx = std::inner_product(
				last_w.begin(), last_w.end(), x.begin(), 0.0f
			);
			sum_y_wx += input.ys[sample] - wx;
		}

		int m = input.xs.size();

		float grad = (m * last_b - sum_y_wx) * 2;
		return grad / input.xs.size();
	}

	float InitializeB(INPUT& input)
	{
		return 0;
	}

	std::vector<float> InitializeW(INPUT& input)
	{
		std::vector<float> w(input.xs[0].size(), 0.1);
		return w;
	}


	int TESTGradientB(INPUT& input)
	{
		const float delta = 0.001;

		std::vector<float> w = InitializeW(input);
		float last_b = InitializeB(input);
		std::vector<float> preds = Predict(input, w, last_b);
		float last_loss = CalMSELoss(input.ys, preds);

		float new_b = last_b + delta;
		preds = Predict(input, w, new_b);
		float new_loss = CalMSELoss(input.ys, preds);

		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cout << __FUNCTION__ << std::endl;
		std::cout << "\t numetric gradient = " << (new_loss - last_loss) / (new_b - last_b) << std::endl;


		float sym_grad_last_b = CalGradientB(input, w, last_b);
		float sym_grad_new_b = CalGradientB(input, w, new_b);
		std::cout << "\t symbol gradient(last) = " << sym_grad_last_b << std::endl;
		std::cout << "\t symbol gradient(new) = " << sym_grad_new_b << std::endl;
		return 0;
	}

	int TESTGradientW(INPUT& input)
	{
		const float delta = 0.001; //足够小的一个正数
		const int test_dim = 0; //w是一个向量，我们选择一个维度做验证

		assert(test_dim >= 0 && test_dim < input.xs[0].size());

		std::vector<float> last_w = InitializeW(input);
		float b = InitializeB(input) + 0.1;
		std::vector<float> preds = Predict(input, last_w, b);
		float last_loss = CalMSELoss(input.ys, preds);//在一个点上计算loss(输入是高维数据，这就是高维空间中的一个点)

		std::vector<float> new_w;
		std::copy(last_w.begin(), last_w.end(), std::back_inserter(new_w));
		//std::transform(last_w.begin(), last_w.end(), std::back_inserter(new_w), [delta](float val) { return val + delta; });
		new_w[test_dim] += delta; //测试维度上的值做微小的偏移
		preds = Predict(input, new_w, b);
		float new_loss = CalMSELoss(input.ys, preds); //在偏移后的点上计算loss

		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cout << __FUNCTION__ << std::endl;
		std::cout << "\t numetric gradient = " << (new_loss - last_loss) / (delta) << std::endl; //数值梯度


		std::vector<float> sym_grad_last = CalGradientW(input, last_w, b); //第一点的解析梯度
		std::vector<float> sym_grad_new = CalGradientW(input, new_w, b); //偏移后的点的解析梯度
		std::cout << "\t symbol gradient(last) = " << sym_grad_last[test_dim] << std::endl;
		std::cout << "\t symbol gradient(new) = " << sym_grad_new[test_dim] << std::endl;
		//数值梯度和两个解析梯度的值应该是接近的
		return 0;
	}

	float GetCurrentLR(int epoch, int epoch_total)
	{
		const float lr = 0.5f;
		return lr;
	}



	int GradientDecent(INPUT& input, OUTPUT& output)
	{
		//验证梯度代码
		TESTGradientB(input);
		TESTGradientW(input);

		const int epoch_total = 100;

		int m = input.xs.size();
		int dim = input.xs[0].size();
		std::vector<float> last_w = InitializeW(input);
		float last_b = InitializeB(input);
		for (int epoch = 0; epoch < epoch_total; epoch++)
		{
			float lr = GetCurrentLR(epoch, epoch_total);
			float grad_b = CalGradientB(input, last_w, last_b);
			std::vector<float> grad_w = CalGradientW(input, last_w, last_b);

			float new_b = last_b - lr * grad_b;


			std::vector<float> gradw_mul_lr;
			std::transform(grad_w.begin(), grad_w.end(), std::back_inserter(gradw_mul_lr), [lr](float val) { return val * lr;  });
			VEC_OP<float> last_w_op(last_w);
			std::vector<float> new_w = last_w_op - gradw_mul_lr;

			last_b = new_b;
			for (int k = 0; k < last_w.size(); k++) last_w[k] = new_w[k];


			std::vector<float> preds = Predict(input, last_w, last_b);
			float loss = CalMSELoss(preds, input.ys);
			if ((epoch + 1) % 10 == 0)
				std::cout << "epoch: " << epoch + 1 << ", loss:" << loss << std::endl;
		}

		output.b = last_b;
		output.w.clear();
		std::copy(last_w.begin(), last_w.end(), std::back_inserter(output.w));
		return 0;

	}

}
