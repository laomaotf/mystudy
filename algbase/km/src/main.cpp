#include <iostream>
#include <vector>
#include <string>
#include<set>
#include "opencv2/opencv.hpp"

class OBJ
{
public:
	OBJ()
	{
		m_conn= -1;
		m_visited = false;
	}
	~OBJ()
	{

	}
	OBJ(const OBJ& obj)
	{
		m_conn = -1;
		m_visited = false;
		m_conn = obj.m_conn;
		m_visited = obj.m_visited;
	}
public:
	OBJ& operator=(const OBJ& obj)
	{
		m_conn = -1;
		m_visited = false;
		m_conn= obj.m_conn;
		m_visited = obj.m_visited;
		return *this;
	}
public:
	bool Visited()
	{
		return m_visited;
	}
	int Matched()
	{
		return m_conn;
	}
	void SetVisited(bool val)
	{
		m_visited = val;
	}
	void SetMatched(int index)
	{
		m_conn = index;
	}
private:
	bool m_visited;
	int m_conn;
};

class MESSAGE
{
	cv::Mat m_conn;
	std::vector<OBJ> m_objs_M, m_objs_N;
public:
	MESSAGE()
	{
		m_objs_M.clear();
		m_objs_N.clear();
	}
	~MESSAGE()
	{
	}
	MESSAGE(cv::Mat& conn)
	{
		m_conn = conn.clone();
		m_objs_M.resize(conn.rows);
		m_objs_N.resize(conn.cols);
	}
	MESSAGE(const MESSAGE& msg)
	{
		m_conn = msg.m_conn.clone();
		std::copy(msg.m_objs_M.begin(), msg.m_objs_M.end(), std::back_inserter(m_objs_M));
		std::copy(msg.m_objs_N.begin(), msg.m_objs_N.end(), std::back_inserter(m_objs_N));

	}
public:
	cv::Mat show(std::string name = "canvas",int sec = -100)
	{
		int unit = 64;
		int mn_radius = unit / 4;
		int bnd_size = unit * 2;
		int sep_size = unit * 4;
		int width = 2 * unit + 2 * bnd_size + sep_size;
		int height = std::max<int>(m_conn.cols, m_conn.rows) * unit + 2 * bnd_size;

		cv::Mat canvas = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
		canvas += CV_RGB(255, 255, 255);

		for (int m = 0; m < m_objs_M.size(); m++)
		{
			OBJ& obj = m_objs_M[m];
			int n = obj.Matched();
			if (n < 0)
				continue;
			cv::Point pm(bnd_size, m * unit + bnd_size);
			cv::Point pn(bnd_size + sep_size, n * unit + bnd_size);
			cv::line(canvas, pm, pn, CV_RGB(255, 255, 0), 3);
		}

		for (int m = 0; m < m_conn.rows; m++)
		{
			cv::Point pm(bnd_size, m * unit + bnd_size);
			for (int n = 0; n < m_conn.cols; n++)
			{
				cv::Point pn(bnd_size + sep_size, n * unit + bnd_size);
				if (m_conn.at<int>(m, n) == 1)
				{
					cv::line(canvas, pm, pn, CV_RGB(0, 255, 0), 1);
				}
				cv::circle(canvas, pn, mn_radius, CV_RGB(0, 0, 255), -1);
			}
			cv::circle(canvas, pm, mn_radius, CV_RGB(0, 0, 255/2), -1);
		}
		cv::imshow(name, canvas);
		cv::waitKey(sec);
		return canvas;
	}
public:
	MESSAGE& operator=(const MESSAGE& msg)
	{
		m_conn = msg.m_conn.clone();
		m_objs_M.clear();
		m_objs_N.clear();
		std::copy(msg.m_objs_M.begin(), msg.m_objs_M.end(), std::back_inserter(m_objs_M));
		std::copy(msg.m_objs_N.begin(), msg.m_objs_N.end(), std::back_inserter(m_objs_N));
		return *this;
	}
	void ClearVisitedAll()
	{
		for (auto& obj : m_objs_M)
		{
			obj.SetVisited(false);
		}
		for (auto& obj : m_objs_N)
		{
			obj.SetVisited(false);
		}
	}
	bool Connected(int m, int n)
	{
		return m_conn.at<int>(m,n) > 0 ? true : false;
	}
	bool VisitedN(int n)
	{
		return m_objs_N[n].Visited();
	}
	bool VisitedM(int m)
	{
		return m_objs_M[m].Visited();
	}
	void AddExtend(int m, int n)
	{
		m_objs_N[n].SetMatched(m);
		m_objs_M[m].SetMatched(n);
	}
	void SetVisited(int m, int n)
	{
		if(m >= 0)
		m_objs_M[m].SetVisited(true);
		if(n >= 0)
		m_objs_N[n].SetVisited(true);
	}
	void RemoveExtend(int m, int n)
	{
		if(m >= 0)
			m_objs_M[m].SetMatched(-1);
		if(n >= 0)
			m_objs_N[n].SetMatched(-1);
	}
	bool InExtendPathM(int m)
	{
		return m_objs_M[m].Matched() >= 0;
	}
	bool InExtendPathN(int n)
	{
		return m_objs_N[n].Matched() >= 0;
	}
	int GetMatchedM(int n)
	{
		return m_objs_N[n].Matched();
	}

	int GetMatchedN(int m)
	{
		return m_objs_N[m].Matched();
	}
};

bool HungarianTryM(int m, int N, MESSAGE& msg)
{
	/*
	* 为m搜索一个匹配的n: 遍历所有的N
	* s1. n和m连通，(m,n)不属于当前extend path集合，而且n不属于目前的extend path中顶点, 则直接把(m,n)加入extend path，终止对m的匹配
	* s2. n和m连通, (m,n)不属于当前extend path集合，但是n属于目前的extend path中的顶点，而且n原本的匹配点m_old尚未在本轮中探索过，则把(m,n)加入extend path，把(m_old,n)移出extend path，尝试为m_old再选择一个匹配点
	* s3. 如果n无法满足1或2，则跳过n，考虑用n+1和m匹配
	*/
	cv::Mat canvas = msg.show("tryM-input");
	for (int n = 0; n < N; n++)
	{
		if (!msg.Connected(m, n))
			continue;
		if (!msg.InExtendPathN(n))
		{//s1
			msg.AddExtend(m, n);
			msg.SetVisited(m,n);
			canvas = msg.show("update aug path");
			return true;
		}
		//s2
		int m_old = msg.GetMatchedM(n);
		if (msg.VisitedM(m_old))
			continue; //n is matched with m_old but m_old is explored already in this round
		MESSAGE msg0(msg);
		msg0.AddExtend(m, n);
		msg0.RemoveExtend(m_old, -1);
		msg0.SetVisited(m,n);
		if (HungarianTryM(m_old, N, msg0))
		{
			msg = msg0; //keep success try
			canvas = msg.show("update aug path");
			return true;
		}
	}
	return false;
}


bool Hungarian(int M, int N, std::vector<std::pair<int,int>>& connected)
{
	cv::Mat conn = cv::Mat::zeros(cv::Size(N, M), CV_32SC1);

	for (auto& pair : connected)
	{
		int m = pair.first, n = pair.second;
		conn.at<int>(m, n) = 1;
	}
	MESSAGE msg(conn);
	msg.show("init");
	int num_matched = 0;
	for (int m = 0; m < M; m++)
	{
		msg.ClearVisitedAll();
		if (HungarianTryM(m,N,msg))
			num_matched++;
		msg.show("final" + std::to_string(m));
	}
	msg.show("final");
	return 0;
}


int main(int argc, char* argv)
{
	int M = 3, N = 3;
	std::vector<std::pair<int, int>> conn;
	conn.push_back(std::make_pair(0, 0));
	conn.push_back(std::make_pair(0, 1));
	conn.push_back(std::make_pair(1, 0));
	conn.push_back(std::make_pair(1, 2));
	conn.push_back(std::make_pair(2, 0));
	Hungarian(M, N, conn);
	return 0;
}