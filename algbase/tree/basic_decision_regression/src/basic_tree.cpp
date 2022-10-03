#include "basic_tree.h"
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <string>
#include <sstream>
#include <set>
#include "utils.h"
namespace basic_tree
{
	void CTree::Train(std::vector<std::vector<VALUE>>& X, CTrainParam& param)
	{
		int feat_num = X[0].size();
		int sample_num = X.size();

		std::vector<CNode*> open_nodes;
		CNode* cur_node = create_node();
		cur_node->_parent = NULL; //a-set parent node
		cur_node->_indices.clear(); //b-set indices
		cur_node->_depth = 0; //depth
		for (int k = 0; k < sample_num; k++)
		{
			cur_node->_indices.push_back(k);
		}
		cur_node->_class_probs = CalcPred(X, cur_node->_indices); //c-set prediction
		open_nodes.push_back(cur_node);

		while (!open_nodes.empty())
		{
			cur_node = open_nodes.back();
			open_nodes.pop_back();

			if (try_stop_split(X, *cur_node, param.max_depth, param.min_std))
			{
				continue;
			}

			//选择合适的feature 并生成children
			std::vector<float> losses_of_feat = { 1e20 };
			std::vector<int> losses_of_dtype = { 0 };
			std::vector<float> thresholds = { 0 };
			for (int feat_index = 1; feat_index < feat_num; feat_index++)
			{
				std::vector< float > rets = CalcLoss(X, cur_node->_indices, feat_index, param.loss_type);
				if (rets.empty()) std::cout << "ERROR: CalcLoss() return {}" << std::endl;
				float loss_val = rets[0];
				losses_of_feat.push_back(loss_val);

				if (rets.size() == 2)
				{
					losses_of_dtype.push_back(1); //连续值
					thresholds.push_back(rets[1]);
				}
				else
				{
					losses_of_dtype.push_back(0);//离散值
					thresholds.push_back(0);//占位符
				}
			}
			std::vector<int> indices;
			for (int k = 0; k < losses_of_feat.size(); k++) indices.push_back(k);

			std::sort(indices.begin(), indices.end(), [&losses_of_feat](int a, int b) { return losses_of_feat[a] < losses_of_feat[b];  });

			int feat_selected = indices[0];
			float threshold = thresholds[feat_selected];
			cur_node->_feat_index = feat_selected;
			cur_node->_feat_dtype = X[0][feat_selected].dtype();
			cur_node->_threshold = threshold;
			std::map<int, std::vector<int>> splits = SplitX(X, cur_node->_indices, feat_selected, threshold);

			for (auto& itr : splits)
			{
				int c = itr.first;
				std::vector<int>& indices = itr.second;
				CNode* new_node = create_node();
				new_node->_parent = cur_node;
				new_node->_indices = indices;
				new_node->_depth = cur_node->_depth + 1;
				new_node->_class_probs = CalcPred(X, new_node->_indices);
				open_nodes.push_back(new_node);

				cur_node->_children.insert( //set children
					std::make_pair(c, new_node)
				);
			}
		}

		return;
	}

	std::vector<std::map<int, float>>  CTree::Evaluate(std::vector<std::vector<VALUE>>& X)
	{
		int target_dtype = X[0][0].dtype(); //第0维需要存在
		//1--find root
		CNode* root = NULL;
		for (auto one : _nodes)
		{
			if (one->_parent == NULL)
			{
				root = one;
				break;
			}
		}

		//2--evaluation
		std::vector<std::map<int, float>> preds;
		for (auto sample : X)
		{
			//traveling in tree
			CNode* cur_node = root;
			while (1)
			{
				int feat_dtype = cur_node->_feat_dtype;
				int feat_index = cur_node->_feat_index;
				bool find = false;
				if (feat_dtype == 0)
				{
					int c = sample[feat_index].i();
					auto& class2child = cur_node->_children.find(c);
					if (class2child != cur_node->_children.end())
					{
						cur_node = class2child->second;
						find = true;
					}
				}
				else
				{
					float val = sample[feat_index].f();
					if (val <= cur_node->_threshold)
					{
						cur_node = cur_node->_children[0];
					}
					else
					{
						cur_node = cur_node->_children[1];
					}
					find = true;
				}

				if (find == false || cur_node->_children.empty())
				{//node中没有对应的val 或 已经达到叶子节点
					preds.push_back(cur_node->_class_probs);
					break;
				}
			}
		}
		return preds;
	}
};