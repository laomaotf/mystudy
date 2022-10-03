#include "utils.h"

#define LOSS_EPS (1e-5)

namespace loss_func
{
	float SafeLog2(float val)
	{
		if (val < LOSS_EPS) val = LOSS_EPS;
		return log2(val);
	}
	float ENTROPY(std::vector< std::vector<basic_tree::VALUE> >& data, std::vector<int>& indices)
	{
		/////////////////////////////////////////////
		//ͳ��target feat����ȡֵ�ĸ���
		std::map<int, int> counts;
		for (auto index : indices)
		{
			int feat = data[index][0].i(); //Ĭ�ϵ�0ά��Ŀ��
			auto ptr = counts.find(feat);
			if (ptr == counts.end())
				counts.insert(std::make_pair(feat, 1));
			else
				ptr->second++;
		}
		int total = data.size();
		std::vector<float> probs;
		for (auto ptr = counts.begin(); ptr != counts.end(); ptr++)
		{
			probs.push_back(ptr->second / float(total));
		}

		//////////////////////////////////////////////////////
		//������
		std::vector<float> en;
		std::transform(probs.begin(), probs.end(), std::back_inserter(en), [](float val) { return val * SafeLog2(val); });

		return -1 * std::accumulate(en.begin(), en.end(), 0.0f);
	}

	float MSE(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices)
	{//Ŀ��ֵ������ֵ
		float sum_x = 0, sum_xx = 0;
		for (auto& index : indices)
		{
			float val = data[index][0].f();//Ĭ�ϵ�0ά��Ԥ��Ŀ��
			sum_x += val;
			sum_xx += val * val;
		}
		float mean_x = sum_x / indices.size();
		float mean_xx = sum_xx / indices.size();
		float var_x = mean_xx - mean_x * mean_x;

		return var_x; //varԽСԽ��
	}

	float GINI(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices)
	{//gini: Ŀ��ֵ ����ɢֵ
		//GINI ��ӳ�����ȵ�һ��ָ�꣬ԽСԽ����
		//ͳ��Ŀ�����ȡֵ�ĸ���
		std::map<int, int> counts;
		for (auto& index : indices)
		{
			int val = data[index][0].i();//Ĭ�ϵ�0ά��Ԥ��Ŀ��
			auto ptr = counts.find(val);
			if (ptr == counts.end())
				counts.insert(std::make_pair(val, 1));
			else
				ptr->second++;
		}
		int total = data.size();
		std::vector<float> probs;
		for (auto ptr = counts.begin(); ptr != counts.end(); ptr++)
		{
			probs.push_back(ptr->second / float(total));
		}

		//////////////////////////////////////////////////////
		//����gini
		std::vector<float> gini;
		std::transform(probs.begin(), probs.end(), std::back_inserter(gini), [](float pr) { return pr * (1 - pr); });
		return std::accumulate(gini.begin(), gini.end(), 0.0f);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* ������ɢ�����������ݼ�
*/
std::vector<float> CalcLoss_Category(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, std::string loss_type)
{
	std::map<int, std::vector<int> > splits;

	int target_dtype = data[0][0].dtype();

	/////////////////////////////////////////////////
	//��feat��ֵ���data
	for (auto index : indices)
	{
		int val = data[index][feat_index].i();
		auto ptr = splits.find(val);
		if (ptr == splits.end())
		{
			std::vector<int>  init = { index };
			splits.insert(std::make_pair(val, init));
		}
		else
		{
			ptr->second.push_back(index);
		}
	}

	//����loss
	float loss = 0;
	if (target_dtype == 0)
	{//��������
		if (loss_type == "gini")
		{
			for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
			{
				int split_size = ptr->second.size();
				float pr = split_size / (float)(indices.size());
				float gini = loss_func::GINI(data, ptr->second);
				loss += pr * gini;
			}
		}
		else if (loss_type == "id3" || loss_type == "c4.5")
		{
			float en_init = loss_func::ENTROPY(data, indices);
			float en_now = 0;
			for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
			{
				int split_size = ptr->second.size();
				float en = loss_func::ENTROPY(data, ptr->second);
				en_now += en;
			}
			float en_gain = (en_init - en_now);
			if (loss_type == "id3")
			{//��Ϣ����Խ��Խ��
				loss = -en_gain;
			}
			else if (loss_type == "c4.5")
			{//��Ϣ�����Խ��Խ��
				loss = 1 - en_gain / en_init;
			}
		}
	}
	else if (target_dtype == 1)
	{//�ع�����
		float w_total = 0;
		for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
		{
			float w = ptr->second.size();
			float mse = loss_func::MSE(data, ptr->second);
			loss += mse * w;
			w_total += w;
		}
		loss /= w_total;
	}

	return { loss };
}

/*
* ��������ֵ�����������ݼ�
*/
std::vector<float> CalcLoss_Regression(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, std::string loss_type)
{
	std::map<int, std::vector<int> > splits;

	int target_dtype = data[0][0].dtype();

	/////////////////////////////////////////////////
	//����ֵ��������,С����ֵ����Ϊclass 0, ��֮��Ϊclass 1
	std::vector<float> values;
	for (auto index : indices)
	{
		float val = data[index][feat_index].f();
		values.push_back(val);
	}

	std::sort(values.begin(), values.end());
	std::vector<float> thresholds;
	for (int k = 0; k < values.size() - 1; k++)
	{
		if (values[k + 1] - values[k] < LOSS_EPS) continue;
		thresholds.push_back(0.5 * (values[k] + values[k + 1]));
	}

	if (thresholds.empty()) return { 1e9, 1e9 }; //������ֵû�б仯�����ʺ�����������

	float min_loss = 1e9;
	int best_th_index = 0;
	for (int index_th = 0; index_th < thresholds.size(); index_th++)
	{
		std::map<int, std::vector<int>> splits;
		for (auto index : indices)
		{
			float val = data[index][feat_index].f();
			int c = 1;
			if (val <= thresholds[index_th])
				c = 0;

			auto& itr = splits.find(c);
			if (itr == splits.end())
			{
				std::vector<int> init = { index };
				splits.insert(std::make_pair(c, init));
			}
			else
			{
				itr->second.push_back(index);
			}
		}

		//����loss
		float loss = 0;
		if (target_dtype == 0)
		{//��������
			if (loss_type == "gini")
			{
				for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
				{
					int split_size = ptr->second.size();
					float pr = split_size / (float)(indices.size());
					float gini = loss_func::GINI(data, ptr->second);
					loss += pr * gini;
				}
			}
			else if (loss_type == "id3" || loss_type == "c4.5")
			{
				float en_init = loss_func::ENTROPY(data, indices);
				float en_now = 0;
				for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
				{
					int split_size = ptr->second.size();
					float en = loss_func::ENTROPY(data, ptr->second);
					en_now += en;
				}
				float en_gain = (en_init - en_now);
				if (loss_type == "id3")
				{//��Ϣ����Խ��Խ��
					loss = -en_gain;
				}
				else if (loss_type == "c4.5")
				{//��Ϣ�����Խ��Խ��
					loss = 1 - en_gain / en_init;
				}
			}
		}
		else if (target_dtype == 1)
		{//�ع�����
			float w_total = 0;
			for (auto ptr = splits.begin(); ptr != splits.end(); ptr++)
			{
				float w = ptr->second.size();
				float mse = loss_func::MSE(data, ptr->second);
				loss += mse * w;
				w_total += w;
			}
			loss /= w_total;
		}

		if (loss < min_loss || index_th == 0)
		{
			min_loss = loss;
			best_th_index = index_th;
		}
	}

	return { min_loss,thresholds[best_th_index] };
}

std::vector<float> CalcLoss(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, std::string loss_type)
{//����{loss}������{loss,threshold}
	int dtype = data[0][feat_index].dtype();

	if (dtype == 0)
	{
		return CalcLoss_Category(data, indices, feat_index, loss_type);
	}
	else if (dtype == 1)
	{
		return CalcLoss_Regression(data, indices, feat_index, loss_type);
	}
	return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<int, std::vector<int>> SplitX_Category(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index)
{
	std::map<int, std::vector<int> > splits;
	/////////////////////////////////////////////////
	//��feat��ֵ���data
	for (auto index : indices)
	{
		int val = data[index][feat_index].i();
		auto ptr = splits.find(val);
		if (ptr == splits.end())
		{
			std::vector<int>  init = { index };
			splits.insert(std::make_pair(val, init));
		}
		else
		{
			ptr->second.push_back(index);
		}
	}

	return splits;
}

std::map<int, std::vector<int>> SplitX_Regression(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, float threshold)
{
	std::map<int, std::vector<int> > splits;
	/////////////////////////////////////////////////
	//������threshold����0����������1
	for (auto index : indices)
	{
		int c = 1;
		float val = data[index][feat_index].f();
		if (val <= threshold)
			c = 0;
		auto ptr = splits.find(c);
		if (ptr == splits.end())
		{
			std::vector<int>  init = { index };
			splits.insert(std::make_pair(c, init));
		}
		else
		{
			ptr->second.push_back(index);
		}
	}

	return splits;
}

std::map<int, std::vector<int>> SplitX(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, float threshold)
{
	/*
	* ����(class_id, indices)
	* threshold ֻ��regression����
	*/
	std::map<int, std::vector<int>> splits;
	int dtype = data[0][feat_index].dtype();
	if (dtype == 0)
	{
		return SplitX_Category(data, indices, feat_index);
	}
	else if (dtype == 1)
	{
		return SplitX_Regression(data, indices, feat_index, threshold);
	}
	return splits;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::map<int, float> CalcPred(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices)
{
	int target_dtype = data[0][0].dtype();

	std::map<int, float> class_probs;
	if (target_dtype == 0)
	{//category
		for (auto index : indices)
		{
			int c = data[index][0].i();
			auto ptr = class_probs.find(c);
			if (ptr == class_probs.end())
				class_probs.insert(std::make_pair(c, 1));
			else
				ptr->second += 1.0f;
		}

		for (auto& itr : class_probs)
		{
			itr.second /= indices.size();
		}
	}
	else if (target_dtype == 1)
	{//regression
		float avg = 0.0f;
		for (auto index : indices)
		{
			avg += data[index][0].f();
		}
		class_probs.insert(std::make_pair(0, avg / indices.size())); //�ع�����
	}
	return class_probs;
}

//////////////////////////////////////////////////////////////////////////////
bool try_stop_split(std::vector<std::vector<basic_tree::VALUE>>& data, basic_tree::CNode& node, int max_depth, float min_std)
{
	if (node._depth >= max_depth) return true;

	if (node._indices.size() < 2) return true;

	int target_dtype = data[0][0].dtype();

	if (target_dtype == 0)
	{//��ɢ
		std::map<int, int> counter;
		for (auto index : node._indices)
		{
			int c = data[index][0].i();
			auto ptr = counter.find(c);
			if (ptr == counter.end())
			{
				counter.insert(std::make_pair(c, 1));
			}
			else
			{
				ptr->second += 1;
			}
		}

		if (counter.size() < 2) return true;
	}
	else if (target_dtype == 1)
	{//�ع�
		float avg_x = 0, avg_xx = 0;
		for (auto index : node._indices)
		{
			float val = data[index][0].f();
			avg_x += val;
			avg_xx += val * val;
		}
		avg_x /= node._indices.size();
		avg_xx /= node._indices.size();
		float std = std::sqrt(avg_xx - avg_x * avg_x);
		if (std <= min_std)
			return true;
	}
	return false;
}