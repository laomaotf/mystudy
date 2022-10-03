#pragma once
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <iterator>
namespace basic_tree
{
	/*
	DT具有良好的可解释性，可以作为规则挖掘工具: 从一组人工规则中提取一个决策树。最简单情况下，人工规则就是
	利用if - else组织的规则，这时DT相当于重新把if - else组织成树状

	DT处理连续值：取连续两个值的中值做阈值，划分样本
	DT做回归：Loss变成MSE

	*/

	class VALUE
	{
		int m_class;
		float m_value;
		int m_dtype; //0：类别值，1：连续值
	public:
		VALUE()
		{
			m_class = -1;
			m_value = -1;
			m_dtype = -1;
		}
		VALUE(int c)
		{
			m_class = c;
			m_value = -1;
			m_dtype = 0;
		}
		VALUE(float v)
		{
			m_class = -1;
			m_value = v;
			m_dtype = 1;
		}
		~VALUE()
		{
		}
	public:
		VALUE(const VALUE& val)
		{
			m_class = val.m_class;
			m_value = val.m_value;
			m_dtype = val.m_dtype;
		}
		VALUE& operator=(const VALUE& val)
		{
			m_class = val.m_class;
			m_value = val.m_value;
			m_dtype = val.m_dtype;
			return *this;
		}

	public:
		float& f()
		{
			return m_value;
		}
		int& i()
		{
			return m_class;
		}
		int& dtype()
		{
			return m_dtype;
		}
	};
	struct CTrainParam
	{
		std::string loss_type; //id3,c4.5,cart. regression默认使用mse
		float min_std = 0.1;
		int max_depth = 5;
	};

	class CNode
	{
	public:
		std::vector<int> _indices; //train only
	public:
		std::map<int, float> _class_probs;//预测结果，分类问题：(类别，概率)， 回归问题: (0，value)
	public:
		std::map<int, CNode*> _children;
		int _feat_index;
		int _feat_dtype;
		int _depth;
		float _threshold; //只对回归问题有效
		CNode* _parent;
	public:
		CNode()
		{
			_parent = NULL;
			_children.clear();
			_feat_index = -1;
			_feat_dtype = -1;
			_threshold = 0;
		}
		~CNode()
		{
		}
	};

	class CTree
	{
		std::vector< CNode* > _nodes;

	public:
		CTree() { }
		virtual ~CTree()
		{
			for (auto& one : _nodes)
			{
				if (one != NULL)  delete one;
			}
		}
		CTree(const CTree& tree)
		{

		}
	public:
		CNode* create_node()
		{
			CNode* ptr = new CNode();
			_nodes.push_back(ptr);
			return ptr;
		}
		void Train(std::vector<std::vector<VALUE>>& X, CTrainParam& param);
		std::vector<std::map<int, float>>  Evaluate(std::vector<std::vector<VALUE>>& X);
	};
};
