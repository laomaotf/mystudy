#include "lda.h"
#include <iostream>
#include "Eigen/Core"
#include "Eigen/SVD"
namespace lda
{
	struct DATA_PARAM
	{
		std::vector<float> mean;
		std::vector<std::vector<float>> convar;
	};

	std::vector< std::vector<float> > SolveForInv(std::vector< std::vector<float> >& mat_squared, double eps = 1e-6)
	{
		int rows = mat_squared.size();
		int cols = mat_squared[0].size();
		if (rows != cols)
		{
			throw std::runtime_error("only squared mat supported");
		}
		Eigen::MatrixXf J = Eigen::MatrixXf::Zero(rows, cols);
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				J(y, x) = mat_squared[y][x];
			}
		}
		/*
		* J = U*A*V^T
		*/
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::MatrixXf U = svd.matrixU(), V = svd.matrixV(), A = svd.singularValues().asDiagonal();

		//对角矩阵求逆
		Eigen::MatrixXf invA;
		invA.setIdentity(A.rows(), A.cols());
		for (int k = 0; k < A.rows(); k++)
		{
			float val = A(k,k);
			if (fabs(val) > eps)
			{
				invA(k,k) = 1.0f / val;
			}
			else
			{
				invA(k,k) = 0.0f;
			}
		}
	
#if 0
		std::cout << "A:" << A << std::endl;
		std::cout << "U:" << U << std::endl;
		std::cout << "V:" << V << std::endl;
#endif
//		std::cout << "---------" << U * V.transpose() << std::endl;

		/*
		* J = U * A * V^T
		* inv(J) = V * A^(-1) * U^T
		*/
		Eigen::MatrixXf invJ;
		invJ = V * invA * U.transpose();

#if 0
		{
			std::cout << "J:" << std::endl << J << std::endl;
			std::cout << "invJ:" << std::endl << invJ << std::endl;

			Eigen::MatrixXf rebuild = U * A * V.transpose();
			std::cout << "rebuild:" << std::endl << rebuild << std::endl;

			Eigen::MatrixXf res = (rebuild - J);

			std::cout << "rebuild error: " << res.norm() / res.size() << std::endl;


			res = invJ * J;
			Eigen::MatrixXf eye;
			eye.setIdentity(J.rows(), J.cols());
			std::cout << "eye rebuild " << std::endl << res << std::endl;
			res = (res - eye);
			std::cout << "eye error: " << res.norm() << std::endl;
		}
#endif

		std::vector< std::vector<float> > res;
		res.resize(invJ.rows());
		for (int y = 0; y < invJ.rows(); y++)
		{
			res[y].resize(invJ.cols(),0.0f);
			for (int x = 0; x < invJ.cols(); x++)
			{
				res[y][x] = invJ(y, x);
			}
		}


		return res;
	}

	std::shared_ptr< DATA_PARAM > OneClassParam(std::vector< std::vector<float> >& datas)
	{//计算均值和方差
		std::shared_ptr<DATA_PARAM> param = std::shared_ptr<DATA_PARAM>(new DATA_PARAM);

		int num = datas.size();
		int dim = datas[0].size();

		param->mean.resize(dim, 0.0f);
		//1--get means
		for (auto& one : datas)
		{
			if (one.size() != dim)
			{
				throw std::runtime_error("data should be the same dim");
			}
			for (int k = 0; k < one.size(); k++)
			{
				param->mean[k] += one[k];
			}
		}
		for (auto& val : param->mean)
		{
			val /= num;
		}

		//2--remove mean
		std::vector< std::vector<float> > datas_sub_mean;
		for (auto& one : datas)
		{
			std::vector<float> data(dim, 0);
			for (int k = 0; k < dim; k++)
			{
				data[k] = one[k] - param->mean[k];
			}
			datas_sub_mean.push_back(data);
		}

		//3--covariance
		/*
		* C = X^T X / （num-1)
		*/
		std::vector< std::vector<float> >& covar = param->convar;
		covar.resize(dim);
		for (auto& one : covar)
		{
			one.resize(dim);
		}

		std::vector<std::vector<float> > datas_sub_mean_transpose;
		datas_sub_mean_transpose.resize(dim);
		for (auto& one : datas_sub_mean_transpose)
		{
			one.resize(num);
		}

		for (int y = 0; y < dim; y++)
		{
			for (int x = 0; x < num; x++)
			{
				datas_sub_mean_transpose[y][x] = datas_sub_mean[x][y];
			}
		}

		for (int y = 0; y < dim; y++)
		{
			for (int x = 0; x < dim; x++)
			{
				float dot = 0;
				for (int k = 0; k < num; k++)
				{
					dot += datas_sub_mean_transpose[y][k] * datas_sub_mean[k][x];
				}
				covar[y][x] = dot / (num - 1);
			}
		}



		return param;
	}

	std::shared_ptr<MODEL> BuildModel(std::vector<std::vector< std::vector<float> >>& datas)
	{
		int num_classes = datas.size();
		if (num_classes != 2)
		{
			std::runtime_error("only 2 classes supported");
			return NULL;
		}
		int dim = datas[0][0].size();
		std::vector< std::shared_ptr< DATA_PARAM > > params;
		for (int cls = 0; cls < num_classes; cls++)
		{
			std::shared_ptr<DATA_PARAM> param = OneClassParam(datas[cls]);
			params.push_back(param);
		}

		//类内方差
		std::vector< std::vector<float> > covar;
		covar.resize(dim);
		for (int k = 0; k < dim; k++)
		{
			covar[k].resize(dim, 0.0f);
			for (int j = 0; j < dim; j++)
			{
				for (int cls = 0; cls < num_classes; cls++)
					covar[k][j] += params[cls]->convar[k][j];
			}
		}

		//计算S^(-1)
		std::vector< std::vector<float> > inv = SolveForInv(covar);

		/*
		* 计算投影方向w = S^(-1) * mean_diff
		*/
		std::vector<float> mean_diff;
		std::transform(
			params[0]->mean.begin(), params[0]->mean.end(),
			params[1]->mean.begin(),
			std::back_inserter(mean_diff),
			[](float m0, float m1) {  return m0 - m1; }
		);

		std::vector<float> w;
		w.resize(dim, 0.0f);
		for (int k = 0; k < inv.size(); k++)
		{
			std::vector<float> dot;
			std::transform(
				mean_diff.begin(), mean_diff.end(),
				inv[k].begin(),
				std::back_inserter(dot),
				[](float v0, float v1) { return v0 * v1; }
			);
			w[k] = std::accumulate(dot.begin(), dot.end(), 0.0f);
		}


		std::shared_ptr<MODEL> model = std::shared_ptr<MODEL>(new MODEL);
		model->means.resize(num_classes);
		for (int c = 0; c < num_classes; c++)
		{
			std::copy(params[c]->mean.begin(), params[c]->mean.end(), std::back_inserter(model->means[c]));
		}
		std::copy(w.begin(), w.end(), std::back_inserter(model->w));

		return model;

	}

	float Project(std::shared_ptr<MODEL> model, std::vector<float>& data)
	{
		int dim = model->means[0].size();
		if (data.size() != dim)
		{
			throw std::runtime_error("mismatched size for Project()");
		}

		std::vector<float> dot;
		std::transform(
			data.begin(), data.end(),
			model->w.begin(),
			std::back_inserter(dot),
			[](float v0, float v1) { return v0 * v1; }
		);
		return std::accumulate(dot.begin(), dot.end(), 0.0f);
	}

};