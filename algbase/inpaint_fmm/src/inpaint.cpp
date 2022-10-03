#include "stdio.h"
#include "math.h"
#include "inpaint.h"
#include <list>
#include <map>
#include <vector>
using namespace std;

#define KNOWN (1)
#define BAND (2)
#define INSIDE (3)


cv::Mat GetDistanceMap(cv::Mat mask)
{
	cv::Mat dist_map = cv::Mat::zeros(cv::Size(mask.cols+2, mask.rows+2), CV_32SC1);

	int dist_max = mask.cols + mask.rows;

	dist_map = dist_map + dist_max;

	for (int y = 0; y < mask.rows; y++)
	{
		for (int x = 0; x < mask.cols; x++)
		{
			if (mask.at<uchar>(y, x) > 0)
				dist_map.at<int>(y + 1, x + 1) = 0;
		}
	}

	/*
	* forward + backward
	*/
	std::vector<cv::Point> fwd_offsets = { cv::Point(-1,0), cv::Point(-1,-1), cv::Point(0,-1), cv::Point(1,-1) };
	std::vector<cv::Point> bwd_offsets = { cv::Point(1,0), cv::Point(-1,1), cv::Point(0,1), cv::Point(1,1) };
	for(int y = 0; y < mask.rows; y++)
	{
		for(int x = 0; x < mask.cols; x++)
		{

			cv::Point anchor = cv::Point(x + 1, y + 1);
			std::vector<int> neigh_dist;
			for (auto offset : fwd_offsets)
			{
				cv::Point p = cv::Point(anchor.x + offset.x, anchor.y + offset.y);
				neigh_dist.push_back(dist_map.at<int>(p));
			}

			auto m0 = std::min_element(neigh_dist.begin(), neigh_dist.end());

			if (*m0 < dist_map.at<int>(anchor))
			{
				dist_map.at<int>(anchor) = *m0 + 1;
			}
		}
	}

	//2.2--backward scan
	for(int y = mask.rows - 1; y >= 0; y--)
	{
		for(int x = mask.cols - 1; x >= 0; x--)
		{
			cv::Point anchor = cv::Point(x + 1, y + 1);
			std::vector<int> neigh_dist;
			for (auto offset : bwd_offsets)
			{
				cv::Point p = cv::Point(anchor.x + offset.x, anchor.y + offset.y);
				neigh_dist.push_back(dist_map.at<int>(p));
			}
			auto m0 = std::min_element(neigh_dist.begin(), neigh_dist.end());
			if (*m0 < dist_map.at<int>(anchor))
			{
				dist_map.at<int>(anchor) = *m0 + 1;
			}
		}
	}


	return dist_map(cv::Rect(1, 1, mask.cols, mask.rows)).clone();

}


float Solve(cv::Mat F, cv::Mat T, cv::Point p1, cv::Point p2)
{
	float sol = 1e6;

	float t1 = T.at<float>(p1), t2 = T.at<float>(p2);
	if (F.at<int>(p1) == KNOWN)
	{
		if (F.at<int>(p2) == KNOWN)
		{
			float r = std::sqrt(2 - (t1 - t2) * (t1 - t2));
			float s = (t1 + t2 - r) / 2.0f;
			if (s >= t1 && s >= t2) sol = s;
			else
			{
				s += r;
				if (s >= t1 && s >= t2) sol = s;
			}
		}
		else sol = 1 + t1;
	}
	else if (F.at<int>(p2) == KNOWN) sol = 1 + t2;
	return sol;
}
struct DATA
{
	cv::Mat T;
	cv::Mat I;
	cv::Mat F;
	std::multimap<float, cv::Point> Band; //默认按key升序排列


	cv::Mat F_last;
};


DATA* SetupData(cv::Mat I, cv::Mat M, int epsilon)
{
	DATA* data = new DATA;

	I.convertTo(data->I, CV_32FC1, 1 / 255.0, 0);
	data->F = cv::Mat::zeros(I.size(), CV_32SC1);
	data->T = cv::Mat::zeros(I.size(), CV_32FC1);

	///////////////////////////////////////////////////////////////////
	//setup flag F
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			if(M.at<uchar>(y,x) == 0)
				data->F.at<int>(y, x) = KNOWN;
			else
				data->F.at<int>(y, x) = INSIDE;
		}
	}
	cv::Mat M1;
	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(epsilon, epsilon), cv::Point(-1, -1));
	cv::erode(M, M1, kernel, cv::Point(-1, -1), 1, 0);
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			if (M.at<uchar>(y, x) > 0 && M1.at<uchar>(y, x) == 0)
				data->F.at<int>(y, x) = BAND;
		}
	}

//	cv::Mat dist_map = GetDistanceMap(M);
	
	///////////////////////////////////////////////////////////////////
	//setup T
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			if (data->F.at<int>(y, x) == BAND || data->F.at<int>(y, x) == KNOWN)
				data->T.at<float>(y, x) = 0;
			else
				data->T.at<float>(y, x) = 1e6;
		}
	}
	cv::GaussianBlur(data->T, data->T, cv::Size(3, 3), 1.0, 1.0);
	data->F_last = data->T.clone();

	///////////////////////////////////////////////////////////////////
	//setup Band
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			if (data->F.at<int>(y, x) == BAND)
			{
				float t = data->T.at<float>(y, x);
				cv::Point pt(x, y);
				data->Band.insert(std::make_pair(t, pt));
			}
		}
	}


	return data;
}

cv::Mat GetResult(DATA* data)
{
	cv::Mat visI = cv::Mat::zeros(data->I.size(), CV_8UC3);
	if (data->I.channels() == 1)
	{
		for (int y = 0; y < visI.rows; y++)
		{
			for (int x = 0; x < visI.cols; x++)
			{
				int v = data->I.at<float>(y, x) * 255;
				v = std::max<int>(std::min<int>(v, 255), 0);
				visI.at<uchar>(y, x * 3) = v;
				visI.at<uchar>(y, x * 3 + 1) = v;
				visI.at<uchar>(y, x * 3 + 2) = v;
			}
		}
	}
	else
	{
		for (int y = 0; y < visI.rows; y++)
		{
			for (int x = 0; x < visI.cols; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					int v = data->I.at<float>(y, x * 3 + c) * 255;
					v = std::max<int>(std::min<int>(v, 255), 0);
					visI.at<uchar>(y, x * 3 + c) = v;
				}
			}
		}
	}

	return visI;
}

void ShowData(DATA* data)
{
	cv::Mat visF = data->F.clone();
	cv::convertScaleAbs(visF, visF, 50.0, 0.0);
	cv::cvtColor(visF, visF, cv::COLOR_GRAY2BGR);

	int known_num = 0;
	cv::Mat visT = cv::Mat::zeros(data->F.size(), CV_8UC1);
	double m0, m1;
	cv::minMaxLoc(data->T, &m0, &m1);
	m1 = 1000;
	for (int y = 0; y < data->T.rows; y++)
	{
		for (int x = 0; x < data->T.cols; x++)
		{
			float v = data->T.at<float>(y, x);
			if (v > m1) v = m1;
			v = (v - m0) * 255 / (m1 - m0);
			visT.at<uchar>(y, x) = v;

			known_num += data->F.at<int>(y, x) == KNOWN;

			if (data->F.at<int>(y, x) != data->F_last.at<int>(y, x))
				cv::circle(visF, cv::Point(x, y), 3, CV_RGB(255, 0, 0), 1);
		}
	}

	cv::Mat visBand = cv::Mat::zeros(data->F.size(), CV_8UC1);

	for (auto& pair : data->Band)
	{
		visBand.at<uchar>(pair.second.y, pair.second.x) = 255;
	}



	cv::Mat visI = GetResult(data);

	data->F_last = data->F.clone();

	cv::imshow("flag", visF);
	cv::imshow("T", visT);
	cv::imshow("Band", visBand);
	cv::imshow("I", visI);
	cv::waitKey(1);
}
void Inpaint(DATA* data, cv::Point p_ij, int B=1)
{

	if (p_ij.x < 1 || p_ij.y < 1 || p_ij.x >= data->I.cols || p_ij.y >= data->I.rows - 1)
		return;


	float t_ij = data->T.at<float>(p_ij);

	float gx_t_ij = (data->T.at<float>(p_ij.y, p_ij.x + 1) - data->T.at<float>(p_ij.y, p_ij.x - 1)) * 0.5;
	float gy_t_ij = (data->T.at<float>(p_ij.y + 1, p_ij.x) - data->T.at<float>(p_ij.y - 1, p_ij.x)) * 0.5;
	float gnorm_t_ij = std::sqrt(gx_t_ij * gx_t_ij + gy_t_ij * gy_t_ij);
	gx_t_ij /= gnorm_t_ij;
	gy_t_ij /= gnorm_t_ij;

	float Ia = 0, s = 0;
	for (int dy = -B; dy <= B; dy++)
	{
		for (int dx = -B; dx <= B; dx++)
		{
			cv::Point p_kl = cv::Point(p_ij.x + dx, p_ij.y + dy);
			if (p_kl.x < 0 || p_kl.x >= data->I.cols || p_kl.y < 0 || p_kl.y >= data->I.rows)
				continue;
			if (data->F.at<int>(p_kl) != KNOWN) continue;

			float t_kl = data->T.at<float>(p_kl);
			std::vector<float> r = { float(p_ij.x - p_kl.x), float(p_ij.y - p_kl.y) };
			float rnorm = std::sqrt(r[0] * r[0] + r[1] * r[1]);
			r[0] /= rnorm, r[1] /= rnorm;
			float dir = r[0] * gx_t_ij + r[1] * gy_t_ij;
			float dst = 1 / (rnorm * rnorm);
			float lev = 1 / (1 + std::abs<float>(t_ij  - t_kl));
			float w = dir * dst * lev;

			std::vector<float> gradI = { 0.0f, 0.0f };
			if (data->F.at<int>(p_kl.y, p_kl.x + 1) == KNOWN &&
				data->F.at<int>(p_kl.y, p_kl.x - 1) == KNOWN &&
				data->F.at<int>(p_kl.y - 1, p_kl.x) == KNOWN &&
				data->F.at<int>(p_kl.y + 1, p_kl.x) == KNOWN)
			{
				float gx = (data->I.at<float>(p_kl.y, p_kl.x + 1) - data->I.at<float>(p_kl.y, p_kl.x - 1)) * 0.5f;
				float gy = (data->I.at<float>(p_kl.y + 1, p_kl.x) - data->I.at<float>(p_kl.y - 1, p_kl.x)) * 0.5f;
				gradI[0] = gx;
				gradI[1] = gy;
			}

			Ia += w * (data->I.at<float>(p_kl) + gradI[0] * r[0] + gradI[1] * r[1]); //一阶展开式近似
			s += w;
		}
		data->I.at<float>(p_ij) = Ia / s;
	}
	return;
}

cv::Mat DoInpaint(cv::Mat image, cv::Mat mask)
{
	DATA* data = SetupData(image, mask, 3);
	while (!data->Band.empty())
	{
		auto ij = data->Band.begin();
		cv::Point p_ij = ij->second;
		float p_t = ij->first;
		data->Band.erase(ij);

		data->F.at<int>(p_ij) = KNOWN;

		if(1)
		{//remove duplicated points
			std::vector< std::multimap<float, cv::Point>::iterator > keys;
			for(auto itr = data->Band.begin(); itr != data->Band.end(); itr ++)
			{
				if (itr->second.x == p_ij.x && itr->second.y == p_ij.y)
				{
					keys.push_back(itr);
				}
			}
			for (auto one : keys)
			{
				data->Band.erase(one);
			}
		}

		int band_size = data->Band.size();

		std::vector<int> dx = { -1, 0, 1, 0 };
		std::vector<int> dy = { 0, -1, 0, 1 };
		for (int i = 0; i < 4; i++)
		{
			cv::Point p_kl = cv::Point(p_ij.x + dx[i], p_ij.y + dy[i]);
			if (data->F.at<int>(p_kl) != KNOWN)
			{
				if (data->F.at<int>(p_kl) == INSIDE) //无法处理单像素的线
				{
					data->F.at<int>(p_kl) = BAND;
					Inpaint(data, p_kl);
				}
				float t0 = Solve(data->F, data->T, cv::Point(p_kl.x - 1, p_kl.y), cv::Point(p_kl.x, p_kl.y - 1));
				float t1 = Solve(data->F, data->T, cv::Point(p_kl.x + 1, p_kl.y), cv::Point(p_kl.x, p_kl.y - 1));
				float t2 = Solve(data->F, data->T, cv::Point(p_kl.x - 1, p_kl.y), cv::Point(p_kl.x, p_kl.y + 1));
				float t3 = Solve(data->F, data->T, cv::Point(p_kl.x + 1, p_kl.y), cv::Point(p_kl.x, p_kl.y + 1));
				std::vector<float> t = { t0,t1,t2,t3 };
				float t_kl = *std::min_element(t.begin(), t.end());
				data->T.at<float>(p_kl) = t_kl;
				data->Band.insert(std::make_pair(t_kl, p_kl));
			}
		}
		if(data->Band.size() != band_size)
			ShowData(data);
	}
	ShowData(data);
	cv::Mat result = GetResult(data);
	delete data;
	return result;
}

