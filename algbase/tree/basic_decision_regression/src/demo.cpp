#include "basic_tree.h"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
/// 决策树中不同特征选择策略并不会影响最终预测结果，决策树的目的应该只是构造尽量“有效”预测结构，但其预测效果其实是由训练集的代表性决定的。
/// DT只是对特征空间的一次划分，特征选择决定了划分的方法，但是不论是何种划分方法最终特征空间都会被划分完成，所以预测精度是差异不大的

struct DATA_CONFIG
{
	std::string dtype; //value / class
	int pos; //pos
};

std::vector< std::vector<basic_tree::VALUE> > load_data_from_one_file(std::string csv_file, std::map<std::string, DATA_CONFIG>& data_config)
{
	int columns = data_config.size();
	std::vector< std::vector<basic_tree::VALUE> >  data_all;

	std::ifstream fs(csv_file, std::ios::in);
	if (!fs.is_open())
	{
		return data_all;
	}

	std::vector<std::string> col2head;
	char* line_content = new char[2048];
	while (!fs.eof())
	{
		std::vector<basic_tree::VALUE> data(columns);

		line_content[0] = '\0';
		fs.getline(line_content, 2048);
		if (line_content[0] == '\0') break;

		std::string line = line_content;

		std::vector<std::string> splits;
		boost::split(splits, line, boost::is_any_of(","), boost::token_compress_on);
		std::transform(splits.begin(), splits.end(), splits.begin(), [](std::string str) { boost::trim_if(str, boost::is_any_of("\r\n\t ")); return str; });
		if (col2head.empty())
		{
			std::copy(splits.begin(), splits.end(), std::back_inserter(col2head));
			continue; //head
		}

		for (int col = 0; col < splits.size(); col++)
		{
			std::string head = col2head[col];
			auto head_dtype_pair = data_config.find(head);
			if (head_dtype_pair == data_config.end()) continue;
			if (head_dtype_pair->second.dtype == "value")
			{
				basic_tree::VALUE val(boost::lexical_cast<float>(splits[col]));
				data[head_dtype_pair->second.pos] = val;
			}
			else
			{
				basic_tree::VALUE val(boost::lexical_cast<int>(splits[col]));
				data[head_dtype_pair->second.pos] = val;
			}
		}
		data_all.push_back(data);
	}
	delete[] line_content;
	return data_all;
}

int load_data(std::string data_dir,
	std::map<std::string, DATA_CONFIG>& data_config,
	std::vector< std::vector<basic_tree::VALUE> >& train_data, std::vector< std::vector<basic_tree::VALUE> >& test_data)
{
	std::string train_file = data_dir + "/train.csv";
	std::string test_file = data_dir + "/test.csv";

	train_data = load_data_from_one_file(train_file, data_config);
	test_data = load_data_from_one_file(test_file, data_config);

	std::cout << "INFO: train " << train_data.size() << ", test " << test_data.size() << std::endl;
	return 0;
}

void GetDataConfig(std::map<std::string, DATA_CONFIG>& data_config)
{
	data_config.insert(std::make_pair("SalePrice", DATA_CONFIG{ "value",0 })); //默认第一个是目标
	data_config.insert(std::make_pair("YearBuilt", DATA_CONFIG{ "value",1 }));
	data_config.insert(std::make_pair("YearRemodAdd", DATA_CONFIG{ "value",2 }));
	data_config.insert(std::make_pair("Neighborhood", DATA_CONFIG{ "value",3 }));
	data_config.insert(std::make_pair("LotArea", DATA_CONFIG{ "value",4 }));
	data_config.insert(std::make_pair("LotShape", DATA_CONFIG{ "class",5 }));

	data_config.insert(std::make_pair("LotConfig", DATA_CONFIG{ "class",6 }));
	data_config.insert(std::make_pair("HouseStyle", DATA_CONFIG{ "class",7 }));
	data_config.insert(std::make_pair("GarageArea", DATA_CONFIG{ "value",8 }));
	return;
}

void test_tree(basic_tree::CTree& tree, std::vector<std::vector<basic_tree::VALUE>>& X)
{
	std::vector<std::map<int, float>> preds = tree.Evaluate(X);
	if (X[0][0].dtype() == 0)
	{
		float recalling = 0;
		for (int k = 0; k < preds.size(); k++)
		{
			int pred_c = -1;
			float pred_prob = -1;
			for (auto itr : preds[k])
			{
				if (itr.second > pred_prob)
				{
					pred_c = itr.first;
					pred_prob = itr.second;
				}
			}

			if (pred_c == X[k][0].i())
			{
				recalling += 1;
			}
		}
		std::cout << "REC: " << recalling / X.size() << std::endl;
	}
	else if (X[0][0].dtype() == 1)
	{
		float MAPD = 0;
		for (int k = 0; k < preds.size(); k++)
		{
			float err = std::abs(preds[k][0] - X[k][0].f());
			MAPD += err / std::max<float>(1e-5, X[k][0].f());
		}
		std::cout << "MAPD: " << MAPD / X.size() << std::endl;
	}
	return;
}



void test_forest(std::vector<basic_tree::CTree*>& forest, std::vector<std::vector<basic_tree::VALUE>>& X)
{
	std::vector< std::vector<std::map<int, float>> > preds_all;
	std::set<int> output_classes;
	for (auto tree : forest)
	{
		std::vector<std::map<int, float>> preds = tree->Evaluate(X);
		for (auto& predx : preds)
		{
			for (auto& pair : predx)
			{
				output_classes.insert(pair.first);
			}
		}
		preds_all.push_back(preds);
	}

	//soft-bagging
	std::vector<std::vector<float>> probs_all;
	{
		for (int x = 0; x < X.size(); x++)
		{
			std::vector<float> probs;
			std::vector<int> counts;
			probs.resize(output_classes.size(), 0.0f);
			counts.resize(output_classes.size(), 0);
			for (int tree = 0; tree < preds_all.size(); tree++)
			{
				for (auto& pair : preds_all[tree][x])
				{
					int label = pair.first;
					float prob = pair.second;
					probs[label] += prob;
					counts[label] += 1;
				}
			}
			for (int c = 0; c < probs.size(); c++)
			{
				if (counts[c] < 1) continue;
				probs[c] /= counts[c];
			}
			probs_all.push_back(probs);
		}
	}

	if (X[0][0].dtype() == 0)
	{//classification
		float recalling = 0;
		for (int k = 0; k < probs_all.size(); k++)
		{
			int pred_c = -1;
			float pred_prob = -1;
			for(int c = 0; c < probs_all[k].size(); k++)
			{
				if(probs_all[k][c] > pred_prob)
				{
					pred_c = c;
					pred_prob = probs_all[k][c];
				}
			}
			if (pred_c == X[k][0].i())
			{
				recalling += 1;
			}
		}
		std::cout << "REC: " << recalling / X.size() << std::endl;
	}
	else if (X[0][0].dtype() == 1)
	{
		float MAPD = 0;
		for (int k = 0; k < probs_all.size(); k++)
		{
			float err = std::abs(probs_all[k][0] - X[k][0].f());
			MAPD += err / std::max<float>(1e-5, X[k][0].f());
		}
		std::cout << "MAPD: " << MAPD / X.size() << std::endl;
	}

	return;
}

int main_tree()
{
	std::map<std::string, DATA_CONFIG> data_config;
	GetDataConfig(data_config);

	std::vector<std::vector<basic_tree::VALUE>> trainset, testset;
	basic_tree::CTrainParam param;

	std::string root_dir(getenv("DATASET_ROOT_DIR"));
	root_dir += "house_price\\";
	load_data(root_dir, data_config, trainset, testset);
	param.loss_type = "gini";
	param.min_std = 0.1;

	std::vector<int> depths = { 3,10,50 };
	for (auto depth : depths)
	{
		basic_tree::CTree tree_object;
		param.max_depth = depth;
		tree_object.Train(trainset, param);
		// test
		std::cout << "max depth:" << param.max_depth << std::endl;
		std::cout << "-------evaluate trainset----------" << std::endl;
		test_tree(tree_object, trainset);
		std::cout << "-------evaluate testset----------" << std::endl;
		test_tree(tree_object, testset);
	}
	return 0;
}


std::vector<std::vector<basic_tree::VALUE>> resample_dataset(std::vector<std::vector<basic_tree::VALUE>>& dataset, float ratio = 0.5)
{
	std::vector<std::vector<basic_tree::VALUE>>  output;
	for (int k = 0; k < dataset.size(); k++)
	{
		if (  (rand() % 100) / 100.0   > ratio)
		{
			output.push_back(dataset[k]);
		}
	}
	return output;
}

int main_forest(int tree_total, unsigned long seed,float resample_ratio)
{
	srand(seed);
	std::map<std::string, DATA_CONFIG> data_config;
	GetDataConfig(data_config);

	std::vector<std::vector<basic_tree::VALUE>> trainset, testset;
	basic_tree::CTrainParam param;

	std::string root_dir(getenv("DATASET_ROOT_DIR"));
	root_dir += "house_price\\";
	load_data(root_dir, data_config, trainset, testset);
	param.loss_type = "gini";
	param.min_std = 0.1;

	{
		std::vector<basic_tree::CTree*> forest;
		for (int tree_index = 0; tree_index < tree_total; tree_index++)
		{
			std::cout << "training tree " << tree_index + 1 <<"/"<<tree_total << std::endl;
			basic_tree::CTree* tree_object = new basic_tree::CTree();
			std::vector<std::vector<basic_tree::VALUE>>  trainset_down = resample_dataset(trainset,resample_ratio);
			param.max_depth = 10;
			tree_object->Train(trainset_down, param);
			forest.push_back(tree_object);
		}
		// test
		std::cout << "max depth:" << param.max_depth << std::endl;
		std::cout << "-------evaluate trainset----------" << std::endl;
		test_forest(forest, trainset);
		std::cout << "-------evaluate testset----------" << std::endl;
		test_forest(forest, testset);

		for (auto tree : forest)
		{
			delete tree;
		}
	}
	return 0;
}


int main(int argc, char* argv[])
{
	//return main_tree();

	unsigned long seed = (unsigned long)time(0);
	int tree_total = 10;
	float ratio = 0.5f;
	return main_forest(tree_total, seed, ratio);
}
