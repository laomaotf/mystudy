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
	DT�������õĿɽ����ԣ�������Ϊ�����ھ򹤾�: ��һ���˹���������ȡһ�����������������£��˹��������
	����if - else��֯�Ĺ�����ʱDT�൱�����°�if - else��֯����״

	DT��������ֵ��ȡ��������ֵ����ֵ����ֵ����������
	DT���ع飺Loss���MSE

	*/

	class VALUE
	{
		int m_class;
		float m_value;
		int m_dtype; //0�����ֵ��1������ֵ
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
		std::string loss_type; //id3,c4.5,cart. regressionĬ��ʹ��mse
		float min_std = 0.1;
		int max_depth = 5;
	};

	class CNode
	{
	public:
		std::vector<int> _indices; //train only
	public:
		std::map<int, float> _class_probs;//Ԥ�������������⣺(��𣬸���)�� �ع�����: (0��value)
	public:
		std::map<int, CNode*> _children;
		int _feat_index;
		int _feat_dtype;
		int _depth;
		float _threshold; //ֻ�Իع�������Ч
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
