//////////////////////////////////////////////////////////////////////////////
//Author: Awni Hannun
//
//This is an example CTC decoder written in Python.The code is
//intended to be a simple exampleand is not designed to be
//especially efficient.
//
//The algorithm is a prefix beam search for a model trained
//with the CTC loss function.
//
//For more details checkout either of these references :
//https://distill.pub/2017/ctc/#inference
//https://arxiv.org/abs/1408.2873
//////////////////////////////////////////////////////////////////////////////

#include "beam_decode.h"
#include <cmath>
#include <string>
#include <sstream>
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
namespace beam_decoder
{
	const float NEG_INF = (-FLT_MAX);

	class BEAM 
	{
		std::map<std::string, std::vector<float>> _data;
	public:
		BEAM(std::string key, float p_b, float p_nb)
		{
			_data.clear();
			this->insert_or_update(key, p_b, p_nb);
		}
		BEAM()
		{
			_data.clear();
		}
		const BEAM(const BEAM& beam)
		{
			*this = beam;
		}
		~BEAM()
		{
			_data.clear();
		}
	public:
		BEAM& operator=(const BEAM& beam)
		{
			_data.clear();
			for (auto& itr = beam._data.begin(); itr != beam._data.end(); itr++)
			{
				std::string key = itr->first;
				std::vector<float> value;
				std::copy(itr->second.begin(), itr->second.end(), std::back_inserter(value));
				_data.insert(std::make_pair(key, value));
			}
			return *this;
		}
	public:
		std::vector<float> operator[](std::string key)
		{
			auto& itr = _data.find(key);
			if (itr == _data.end())
			{
				return std::vector<float>{NEG_INF, NEG_INF};
			}

			return itr->second;
		}

		int insert_or_update(std::string key, float p_b, float p_nb)
		{
			std::vector<float> p = { p_b, p_nb };
			return insert_or_update(key, p);
		}

		int insert_or_update(std::string key, std::vector<float>& value)
		{
			auto& itr = _data.find(key);
			if (itr != _data.end())
			{
				itr->second.clear();
				std::copy(value.begin(), value.end(), std::back_inserter(itr->second));
			}
			else
			{
				_data.insert(std::make_pair(key, value));
			}
			return 0;
		}

		std::vector<std::string> keys()
		{
			std::vector<std::string> keys_all;
			for (auto itr = _data.begin(); itr != _data.end(); itr++)
			{
				keys_all.push_back(itr->first);
			}
			return keys_all;
		}
	};

	// log sum( exp(x) ) = a + log sum (exp (x - a) )
	//a����������ֵ��������ȡa = max(x)���������
	//�ú�������������ʳ˷�(�������ڸ��ʳ˷�)
	inline float LogSumExp(float a, float b, float c)
	{
		float m = std::max<float>(std::max<float>(a, b), c);
		float sumexp = exp(a - m) + exp(b - m) + exp(c - m);
		return log(sumexp) + m;
	}

	inline float LogSumExp(float a, float b)
	{
		float m = std::max<float>(a, b);
		float sumexp = exp(a - m) + exp(b - m);
		return log(sumexp) + m;
	}


	//beamʹ���ַ�������������(1,2,3,4,5)��Ӧ��beam�ַ���ʽ��"1-2-3-4-5"
	std::vector<int> str2vec(std::string str, std::string sep = "-")
	{
		std::vector<std::string> split_str;
		boost::split(split_str, str, boost::is_any_of(sep),boost::algorithm::token_compress_on);
		std::vector<int> split;
		for (auto& val : split_str)
		{
			split.push_back(
				boost::lexical_cast<int>(val)
			);
		}
		return split;
	}

	std::string vec2str(std::vector<int>& values, std::string sep = "-")
	{
		std::string str;
		for(auto itr = values.begin(); itr != values.end(); itr++)
		{
			std::string s = boost::lexical_cast<std::string>(*itr);
			str += s;
			if (itr + 1 != values.end())
				str += "-";
		}
		return str;
	}

	int FirstPassLM(INPUT& input, OUTPUT& output, int blank)
	{
		int T = input.probs.rows, S = input.probs.cols;
		

		//��ʼ������BEAM
		std::vector<BEAM> beams;
		BEAM beam_init("", 0, NEG_INF);
		beams.push_back(beam_init);

		for (int tick = 0; tick < T; tick++)
		{//����ʱ��T
			BEAM next_beam;//��tickʱ�̣������������ַ���չ���е�beam

			for (int sym = 0; sym < S; sym++)
			{//���������ַ�
				float p = input.probs.at<float>(tick, sym); //���ַ��ĸ���

				for(auto& last_beam : beams)
				{//������չ���е�beam
					std::string prefix = last_beam.keys()[0]; //ֻ�洢һ��beam
					std::vector<float> prefix_probs = last_beam[prefix];
					float p_b = prefix_probs[0], p_nb = prefix_probs[1];
					if (sym == blank)
					{//��ǰʱ���ַ�ȡ�հ��ַ���prefix���䣬����һ����end-with-blank���ʸ���
						std::vector<float> next_probs = next_beam[prefix];
						float n_p_b = next_probs[0], n_p_nb = next_probs[1];
						n_p_b = LogSumExp(n_p_b, p_b + p, p_nb + p); //�����µ�end-with-blank����
						next_beam.insert_or_update(prefix, n_p_b, n_p_nb);
						continue;
					}


					/*��ǰ�ַ����ǿհ��ַ���������չprefix	*/
					int end_t = -1;
					std::vector<int> prefix_int, next_prefix_int;
					if (prefix != "")
					{
						prefix_int = str2vec(prefix);
						end_t = prefix_int.back();
					}
					std::copy(prefix_int.begin(), prefix_int.end(), std::back_inserter(next_prefix_int));
					next_prefix_int.push_back(sym); 
					std::string next_prefix = vec2str(next_prefix_int); //�µ�string
					std::vector<float> next_prob = next_beam[next_prefix]; 
					float n_p_b = next_prob[0], n_p_nb = next_prob[1];


					//���������ַ��Ƿ��ǰһʱ���ַ�һ�£���ֻ����end-with-non-blank����
					if (sym != end_t)
					{//�����ַ���ǰһ���ַ���ͬ
						n_p_nb = LogSumExp(n_p_nb, p_b + p, p_nb + p);
					}
					else
					{//�����ַ���ǰһ���ַ���ͬ���µ�nb���ʲ���Ҫ������ǰ��nb����
						n_p_nb = LogSumExp(n_p_nb, p_b + p);
					}

	
					//���¸���ֵ��������Ե��ø����ӵ�����ģ�ͣ�����beam�ĸ���
					next_beam.insert_or_update(next_prefix, n_p_b, n_p_nb);
					


					//�����ǰ�ַ���ǰһ���ַ���ͬ����Ϊ������next_prefix��prefix���ܶ�Ӧ
					//��ͬ���������������Ҫ������չǰ��beam����
					if (sym == end_t)
					{ //��Ϊblank�Ĵ��ڣ�t-1ʱ�̵����о�����չ��Ľ������һ��
						std::vector<float> next_prob = next_beam[prefix];
						float n_p_b = next_prob[0], n_p_nb = next_prob[1];
						n_p_nb = LogSumExp(n_p_nb, p_nb + p);
						next_beam.insert_or_update(prefix, n_p_b, n_p_nb);
					}
				}
			}

			/*
			* ����next_beam�д洢[0,tick]ʱ��������е�beam��������Ҫ����ѡ������ߵ�һ��beam
			*/
			std::vector< std::pair<std::string, float> > key_prob_list;
			std::vector<std::string> prefixs = next_beam.keys();
			for (auto& prefix : prefixs)
			{
				std::vector<float> prob = next_beam[prefix];
				float p = LogSumExp(prob[0], prob[1]);
				key_prob_list.push_back(std::make_pair(prefix, p));
			}
			
			std::stable_sort(key_prob_list.begin(), key_prob_list.end(),
				[](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {return a.second > b.second;});
		
	
			//����beam
			beams.clear();  
			for (auto key_prob : key_prob_list)
			{
				if (beams.size() >= input.beam_size) break;
				std::string prefix = key_prob.first;
				std::vector<float> prob = next_beam[prefix];
				BEAM beam(prefix, prob[0], prob[1]);
				beams.push_back(beam);
			}

		}

		{
			std::string prefix = beams[0].keys()[0];
			std::vector<float> probs = beams[0][prefix];
			output.prob = -1*LogSumExp(probs[0], probs[1]);
			output.beam = str2vec(prefix);
		}
		return 0;
	}

};