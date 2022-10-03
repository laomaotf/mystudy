
#include "logreg.h"
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <string>
#include <sstream>

namespace logreg
{

	std::vector<float> get_dim(int dim, INPUT& input)
	{
		std::vector<float> one_dim;
		for (auto& one : input.xs)
		{
			one_dim.push_back(one[dim]);
		}
		return one_dim;
	}

	template<class T>
	class VEC_OP
	{
		std::vector<T> _data;
	public:
		VEC_OP(std::vector<T>& data)
		{
			std::copy(data.begin(), data.end(), std::back_inserter(_data));
		}
		~VEC_OP()
		{
			_data.clear();
		}
	public:
		float sum()
		{
			float val = std::accumulate(_data.begin(), _data.end(), 0.0f);
			return val;
		}
		float mean()
		{
			float val = std::accumulate(_data.begin(), _data.end(), 0.0f) / _data.size();
			return val;
		}
		std::vector<T> pow2()
		{
			std::vector<T> output;
			std::transform(_data.begin(), _data.end(), std::back_inserter(output), [](float val) {return val * val;});
			return output;

		}

		std::vector<int> operator==(std::vector<T>& s)
		{
			std::vector<int> output(_data.size(), 0);
			for (int k = 0; k < _data.size(); k++)
			{
				output[k] = (int(s[k]) == int(_data[k])) ? 1 : 0;
			}
			return output;
		}

		std::vector<T> operator-(T s)
		{
			std::vector<T> output;
			std::transform(_data.begin(), _data.end(), std::back_inserter(output), [s](float val) {return val - s; });
			return output;
		}

		std::vector<T> operator-(std::vector<T>& s)
		{
			std::vector<T> output(_data.size(), 0);
			for (int k = 0; k < _data.size(); k++)
			{
				output[k] = _data[k] - s[k];
			}
			return output;
		}

		std::vector<T> operator*(std::vector<T>& s)
		{
			std::vector<T> output(_data.size(), 0);
			for (int k = 0; k < _data.size(); k++)
			{
				output[k] = s[k] * _data[k];
			}
			return output;
		}
	


		std::vector<T> operator+(std::vector<T>& s)
		{
			std::vector<T> output(_data.size(), 0);
			for (int k = 0; k < _data.size(); k++)
			{
				output[k] = s[k] + _data[k];
			}
			return output;
		}
	
	public:
		std::vector<T> data()
		{
			return _data;
		}
	};

	///////////////////////////////////////////////////////////////////////////////////////
	//梯度下降方法求解

	float Sigmoid(float x)
	{
		return 1.0 / (1 + exp(-x));
	}

	float Log(float x, float eps = 0.0001)
	{
		return log(std::max<float>(x, eps));
	}
	std::vector<float> Predict(INPUT& input, std::vector<float> w, float b)
	{
		std::vector<float> preds(input.xs.size(), 0);
		for (int sample = 0; sample < input.xs.size(); sample++)
		{
			std::vector<float>& x = input.xs[sample];
			float y = std::inner_product(x.begin(), x.end(), w.begin(), b);
			preds[sample] = Sigmoid(y);
		}
		return preds;
	}



	float CalAccuracy(std::vector<float>& truths, std::vector<float>& preds, float thresh)
	{
		std::vector<float> preds_label;
		std::transform(preds.begin(), preds.end(), std::back_inserter(preds_label), [thresh](float val) { return val > thresh ? 1.0f : 0.0f; });

		VEC_OP<float> truths_op(truths);
		std::vector<int> hits = (truths_op == preds_label);

		float hit_num = std::accumulate(hits.begin(), hits.end(), 0.0f);
		return hit_num  / truths.size();
	}
	

	float CalCrossEntropy(std::vector<float>& truths, std::vector<float>& preds)
	{
		std::vector<float> log_preds, log_one_sub_preds;
		std::transform(preds.begin(), preds.end(), std::back_inserter(log_preds), [](float val) {return Log(val); });
		std::transform(preds.begin(), preds.end(), std::back_inserter(log_one_sub_preds), [](float val) {return Log(1 - val); });

		std::vector<float> one_sub_truths;
		std::transform(truths.begin(), truths.end(), std::back_inserter(one_sub_truths), [](float val) {return 1 - val; });


		VEC_OP<float> log_preds_op(log_preds);
		std::vector<float> truths_log_preds = log_preds_op * truths;


		VEC_OP<float> one_sub_truths_op(one_sub_truths);
		std::vector<float> one_sub_truths_mul_log_one_sub_preds = one_sub_truths_op * log_one_sub_preds;


		VEC_OP<float> truths_log_preds_op(truths_log_preds);
		std::vector<float> losses = truths_log_preds_op + one_sub_truths_mul_log_one_sub_preds;

		float loss =  std::accumulate(losses.begin(), losses.end(), 0.0f);

		return loss / losses.size();
	}

	float CalcNegLogCrossEntropyLoss(std::vector<float>& truths, std::vector<float>& preds)
	{
		return -1 * CalCrossEntropy(truths, preds);
	}


	std::vector<float> CalGradientW(INPUT& input, std::vector<float>& last_w, float last_b)
	{

		std::vector<float> y_bar = Predict(input, last_w, last_b);

		std::vector<float> gradient(input.xs[0].size());
		int m = input.xs[0].size();
		for (int dim = 0; dim < input.xs[0].size(); dim++)
		{
			std::vector<float> one = get_dim(dim, input);

			VEC_OP<float> one_op(one);

			VEC_OP<float> y_op(input.ys);
			std::vector<float> y_sub_ybar = y_op - y_bar; 

			std::vector<float> grads = one_op * y_sub_ybar;

			gradient[dim] = -1 * std::accumulate(grads.begin(), grads.end(), 0.0f) / input.xs.size();
		}
		return gradient;
	}

	float CalGradientB(INPUT& input, std::vector<float>& last_w, float last_b)
	{
		std::vector<float> y_bar = Predict(input, last_w, last_b);


		VEC_OP<float> y_op(input.ys);
		std::vector<float> y_sub_ybar = y_op - y_bar;

		float grad = std::accumulate(y_sub_ybar.begin(), y_sub_ybar.end(), 0.0f);
		return -1 *  grad / input.xs.size();
	}

	float InitializeB(INPUT& input)
	{
		return 0;
	}

	std::vector<float> InitializeW(INPUT& input, float init = 0.1)
	{
		std::vector<float> w(input.xs[0].size(), init);
		return w;
	}


	int TESTGradientB(INPUT& input)
	{
		const float delta = 0.001;

		std::vector<float> w = InitializeW(input,0.1);
		float last_b = InitializeB(input);
		std::vector<float> preds = Predict(input, w, last_b);
		float last_loss = CalcNegLogCrossEntropyLoss(input.ys, preds);

		float new_b = last_b + delta;
		preds = Predict(input, w, new_b);
		float new_loss = CalcNegLogCrossEntropyLoss(input.ys, preds);

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
		float last_loss = CalcNegLogCrossEntropyLoss(input.ys, preds);//在一个点上计算loss(输入是高维数据，这就是高维空间中的一个点)

		std::vector<float> new_w;
		std::copy(last_w.begin(), last_w.end(), std::back_inserter(new_w));
		//std::transform(last_w.begin(), last_w.end(), std::back_inserter(new_w), [delta](float val) { return val + delta; });
		new_w[test_dim] += delta; //测试维度上的值做微小的偏移
		preds = Predict(input, new_w, b);
		float new_loss = CalcNegLogCrossEntropyLoss(input.ys, preds); //在偏移后的点上计算loss

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
		const float lr = 5.0f / 10.0;
		return lr;
	}



	int Solve(INPUT& input, OUTPUT& output)
	{
		//验证梯度代码
		TESTGradientB(input);
		TESTGradientW(input);

		const int epoch_total = 50;

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
			std::transform(grad_w.begin(), grad_w.end(), std::back_inserter(gradw_mul_lr), [lr](float val) { return val*lr;  });
			VEC_OP<float> last_w_op(last_w);
			std::vector<float> new_w = last_w_op - gradw_mul_lr;

			last_b = new_b;
			for (int k = 0; k < last_w.size(); k++) last_w[k] = new_w[k];

			
			std::vector<float> preds = Predict(input, last_w, last_b);
			float loss = CalcNegLogCrossEntropyLoss(preds, input.ys);
			float acc = CalAccuracy(input.ys, preds);
			if((epoch+1) % 5 == 0)
				std::cout << "epoch: " << epoch + 1 << ", loss:" << loss <<", acc:"<<acc * 100<<"%" << std::endl;
		}

		output.b = last_b;
		output.w.clear();
		std::copy(last_w.begin(), last_w.end(), std::back_inserter(output.w));
		return 0;

	}




};