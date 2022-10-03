#include "pca.h"
#include <iostream>
#include "Eigen/Core"
#include "Eigen/SVD"
namespace pca
{

	std::vector< std::vector<float> > SolveForEigen(std::vector< std::vector<float> >& mat_squared, float R = 0.99)
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
		Eigen::MatrixXf U = svd.matrixU(), V = svd.matrixV(), A = svd.singularValues();
	
		//按照特征值大小，保存足够数量的特征向量
		int keptN = A.size();
		if (R > 0 && R < 1)
		{//A的一个维度为1，按照值从大到小排序
			float total = A.sum();
			float acc = 0;
			int k = 0;
			for (k = 0; k < A.size() && acc < R; k++)
			{
				acc += A(k) / total;
			}
			if(k < A.size()) keptN = k;
		}



#if 0
		Eigen::MatrixXf matA = A.asDiagonal();
		std::cout << U.rows() << "," << U.cols() << std::endl;
		std::cout << V.rows() << "," << V.cols() << std::endl;
		std::cout << A.rows() << "," << A.cols() << std::endl;

		Eigen::MatrixXf rebuild = U * matA * V.transpose();

		Eigen::MatrixXf res = (rebuild - J);

		std::cout << "rebuild error: " << res.norm() << std::endl;
#endif

		std::vector< std::vector<float> > eigvecs;
		eigvecs.resize(keptN);
		for (int y = 0; y <  keptN; y++)
		{
			eigvecs[y].resize(cols);
			for (int x = 0; x < cols; x++)
			{
				eigvecs[y][x] = U(y,x);
			}
		}

		return eigvecs;
	}

	std::shared_ptr<MODEL> BuildModel(std::vector< std::vector<float> >& datas)
	{
		std::shared_ptr<MODEL> model = std::shared_ptr<MODEL>(new MODEL);

		int num = datas.size();
		int dim = datas[0].size();

		model->mean.resize(dim, 0.0f);
		//1--get means
		for (auto& one : datas)
		{
			if (one.size() != dim)
			{
				throw std::runtime_error("data should be the same dim");
			}
			for (int k = 0; k < one.size(); k++)
			{
				model->mean[k] += one[k];
			}
		}
		for (auto& val : model->mean)
		{
			val /= num;
		}

		//2--remove mean
		std::vector< std::vector<float> > datas_sub_mean;
		for (auto& one : datas)
		{
			std::vector<float> data(dim,0);
			for (int k = 0; k < dim; k++)
			{
				data[k] = one[k] - model->mean[k];
			}
			datas_sub_mean.push_back(data);
		}

		//3--covariance
		/*
		* C = X^T X / （num-1)
		*/
		std::vector< std::vector<float> > covar;
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

		std::vector< std::vector<float> > eigvecs = SolveForEigen(covar);
		model->eigvecs.resize(eigvecs.size());
		for(int k = 0; k < eigvecs.size(); k++)
		{
			auto& one = eigvecs[k];
			std::copy(one.begin(), one.end(), std::back_inserter(model->eigvecs[k]));
		}

		return model;
	}

	std::vector<float> Project(std::shared_ptr<MODEL> model, std::vector<float>& data, int N)
	{

		int dim = model->mean.size();
		if (data.size() != model->mean.size())
		{
			throw std::runtime_error("mismatched size for Project()");
		}

		std::vector<float> projs;
		int num = std::min<int>(N, model->eigvecs.size());
		projs.resize(num);
		for (int k = 0; k < num; k++)
		{
			float dot = 0;
			for (int j = 0; j < dim; j++)
			{
				dot += (data[j] - model->mean[j]) * model->eigvecs[k][j];
			}
			projs[k] = dot;
		}
		return projs;
	}

};