#include "stroke_width_transform.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
using namespace std;


#define SWTPI  (3.14159)

class CSWT
{
	cv::Mat m_rayflag, m_edgeflag;
	cv::Mat m_raylen;
	cv::Mat m_sw_data;
	cv::Mat m_norm_dx, m_norm_dy;
	cv::Size m_size;
public:
	CSWT(cv::Size size)
	{
		m_size = size;
		m_rayflag = cv::Mat::zeros(size, CV_32SC1);
		m_edgeflag = cv::Mat::zeros(size, CV_32SC1);
		m_raylen = cv::Mat::zeros(size, CV_32SC1) - 1;
		m_sw_data = cv::Mat::zeros(size, CV_32FC1);
		m_norm_dx = cv::Mat::zeros(size, CV_32FC1);
		m_norm_dy = cv::Mat::zeros(size, CV_32FC1);
	}
	~CSWT()
	{

	}
public:
	int cols()
	{
		return m_size.width;
	}
	int rows()
	{
		return m_size.height;
	}
	int& ray(int row, int col)
	{
		return m_rayflag.at<int>(row, col);
	}
	int& edge(int row, int col)
	{
		return m_edgeflag.at<int>(row, col);
	}
	int& raylen(int row, int col)
	{
		return m_raylen.at<int>(row, col);
	}
	float& sw(int row, int col)
	{
		return m_sw_data.at<float>(row, col);
	}

	float& dx(int row, int col)
	{
		return m_norm_dx.at<float>(row, col);
	}

	float& dy(int row, int col)
	{
		return m_norm_dy.at<float>(row, col);
	}
};



typedef struct
{
	int miny, minx;
	int maxx, maxy;
}SWTLOCATION;



typedef struct
{
	SWTLOCATION location; 
	int label; 
	int area; 
	float sw; //stroke width for this gruop
}SWTSTATS;

struct CSWTGROUP
{
	cv::Mat label_data;
	std::vector<SWTSTATS> groups;
};


int CalcGroupStats(CSWTGROUP* group_data, int total)
{
	int row, col;
	int index;
	
	if(group_data == NULL)
		return -1;
	cv::Mat label_data = group_data->label_data;

	if(total < 1)
		return 0; //done!

	group_data->groups.resize(total);

	for(index = 0; index < total; index++)
	{
		group_data->groups[index].label = index + 1; //assume label is allocated in order and this make following operation is quite easy
		group_data->groups[index].location.minx = group_data->label_data.cols;
		group_data->groups[index].location.miny = group_data->label_data.rows;
		group_data->groups[index].location.maxx = 0;
		group_data->groups[index].location.maxy = 0;
		group_data->groups[index].area = 0;
	}

	for(row = 0; row < group_data->label_data.rows; row++)
	{
		for(col = 0; col < group_data->label_data.cols; col ++)
		{
			SWTLOCATION* pOne;
			int label = label_data.at<int>(row, col);
			if(label < 1)
				continue; //background or something wrong
			group_data->groups[label - 1].area ++;
			pOne = &(group_data->groups[label - 1].location);
			pOne->minx = __min(pOne->minx, col);
			pOne->maxx = __max(pOne->maxx, col);
			pOne->miny = __min(pOne->miny, row);
			pOne->maxy = __max(pOne->maxy, row);

		}
	}
	return 0;

}


inline bool EqualSW(float a, float b, float r0, float r1)
{
	float m0 = std::min<float>(a, b);
	float m1 = std::max<float>(a, b);
	float v = m1 / (m0 + 1);
	if (v <= r1 && v >= r0) return true;
	return false;
}

int GroupSW(CSWT* swt_data, double fMinRatio, double fMaxRatio,
		   CSWTGROUP* group_data) 
{
	int  row,col, index, neighbour_num;
	int* label_pooling;
	int label_total = swt_data->cols() * swt_data->rows(); 
	int next_label;
	int minx, maxx, miny, maxy;
	int group_num;
	int neighbour_labels[4];


	memset(group_data, 0, sizeof(CSWTGROUP));


	int image_area = swt_data->cols() * swt_data->rows();
	int width = swt_data->cols(), height = swt_data->rows();
	minx = 0;
	miny = 0;
	maxx = swt_data->cols() - 1;
	maxy = swt_data->rows() - 1;


	group_data->label_data = cv::Mat::zeros(cv::Size(swt_data->cols(), swt_data->rows()), CV_32SC1);
	cv::Mat& label_data = group_data->label_data;


	//1--scan mask and bulid the label map
	label_pooling = new int[label_total * sizeof(int)];
	if(label_pooling == NULL)
	{
		return -1; 
	}


	next_label = 1; //0 is reserved for background
	for(row = miny; row <= maxy ; row ++)
	{
		for(col = minx; col <= maxx; col++)
		{

			float sw_base = swt_data->sw(row, col);

			if(sw_base < 0)
				continue;



			//1.1--get neighbour pixels' labels
			neighbour_num = 0;

			if( row - 1 >= 0 && col - 1 >= 0 )
			{
				int label = label_data.at<int>(row - 1, col - 1);
				float sw = swt_data->sw(row - 1, col - 1);
				if(EqualSW(sw_base, sw, fMinRatio, fMaxRatio))
				{
					neighbour_labels[neighbour_num] = label;
					neighbour_num ++;
				}
			}

			if(row - 1 >= 0)
			{
				int label = label_data.at<int>(row - 1, col);
				float v = swt_data->sw(row - 1, col);
				if(EqualSW(sw_base, v, fMinRatio, fMaxRatio))
				{
					neighbour_labels[neighbour_num] = label;
					neighbour_num ++;
				}
			}

			if(col + 1 < width && row - 1 >= 0)
			{
				int label = label_data.at<int>(row - 1, col + 1);
				float v = swt_data->sw(row - 1, col + 1);
				if(EqualSW(v, sw_base, fMinRatio, fMaxRatio))
				{
					neighbour_labels[neighbour_num] = label;
					neighbour_num ++;
				}
			}

			if(col - 1 >= 0)
			{
				int label = label_data.at<int>(row, col - 1);
				float v = swt_data->sw(row, col - 1);
				if(EqualSW(v, sw_base, fMinRatio, fMaxRatio))
				{
					neighbour_labels[neighbour_num] = label;
					neighbour_num ++;
				}

			}

			//1.2--update label map and resolve conflict if necessary
			if(neighbour_num == 0)
			{//a new region so allocate a new label for it

				if(next_label < label_total)
				{
					label_data.at<int>(row, col) = next_label;
					label_pooling[next_label] = next_label; //self-mapping
					next_label ++;
				}
				else
				{
					delete[] label_pooling;
					return -1; 			
				}
			}
			else 
			{
				int i,tmp;
				int label_to_search = label_total;
				for(index = 0; index < neighbour_num; index++)
				{//search the minimum root label around current pixel as the merged label
					if(neighbour_labels[index] < 1)
						continue; //-1 or 0 are both impossible


					//find the root label
					i = neighbour_labels[index];
					while (label_pooling[i] != i)
					{
						i = label_pooling[i];
					}

					label_to_search = __min(label_to_search, i);
				}

				//set current with the minimum label
				label_data.at<int>(row, col) = label_to_search;

				//update label map to remove the conflict
				for(index = 0; index < neighbour_num; index ++)
				{
					if(neighbour_labels[index] < 1)
						continue;
					if(neighbour_labels[index] == label_to_search)
						continue; 


					i = neighbour_labels[index];
					while(i != label_to_search && i < next_label) //if the second limitation works something goes wrong
					{//loop is not necessary?
						tmp = label_pooling[i];
						label_pooling[i] = label_to_search;
						i = tmp;
					}
				}
			}


		}
	}


	if(next_label == 1)
	{//not target pixel found
		delete[] label_pooling;
		return -1;
	}



	{
		int new_label = 1;
		for(index = 1; index < next_label; index++)
		{
			if(label_pooling[index] != index)
				continue; //leaf

			label_pooling[index] = -new_label; //negative number indicate it is a new label
			new_label++;

		}
		if(new_label == 1)
		{
			//codes error
			delete[] label_pooling;
			return -1;
		}

		for(index = 1; index < next_label; index ++)
		{
			int nRootLabel;
			if(label_pooling[index] < 1)
				continue; //root

			nRootLabel = label_pooling[index];
			while( nRootLabel < next_label && nRootLabel > 0)
			{
				nRootLabel = label_pooling[nRootLabel];//trace upward for its root
			}
			if(nRootLabel > 0)
			{
				//codes error
				delete[] label_pooling;
				return -1;
			}
			label_pooling[index] = nRootLabel;
		}

		//revert all negative label 
		for(index = 1; index < next_label ; index++)
		{
			if(label_pooling[index] < 0)
				label_pooling[index] = -label_pooling[index];
		}


		for (int y = 0; y < label_data.rows; y++)
		{
			for (int x = 0; x < label_data.cols; x++)
			{
				int label = label_data.at<int>(y, x);
				if (label < 1)
					continue; //background
				label_data.at<int>(y, x) = label_pooling[label];
			}
		}
		group_num = new_label - 1; //used later;

	}



	if(CalcGroupStats(group_data, group_num) != 0)
	{
		delete[] label_pooling;
		return -1;
	}




	delete[] label_pooling;

	return 0;
}



int TextFilter(CSWT* swt_data, CSWTGROUP* group_data)
{

	float text_min_size = 100;
	float sw_max_var = 100;
	float text_min_h = 10;
	float text_max_h = 300;

	float text_max_aspect = 10;



	float text_max_size2sw = 30;

	int index;
	int row,col;

	float* sumx, *sumxx;


	sumx = new float[group_data->groups.size()];
	sumxx = new float[group_data->groups.size()];
	memset(sumx, 0, group_data->groups.size() * sizeof(float));
	memset(sumxx, 0, group_data->groups.size() * sizeof(float));

	cv::Mat label_data = group_data->label_data;

	{

		vector<float> sw_list;

		for(row = 0; row < swt_data->rows(); row++)
		{
			for(col = 0; col < swt_data->cols(); col++)
			{
				float sw = swt_data->sw(row, col);
				int id = label_data.at<int>(row, col);
				if(id < 1)
					continue;


				sumx[id - 1] += sw;
				sumxx[id - 1] += sw * sw;

			}
		}

		for(index = 0; index < group_data->groups.size(); index++)
		{
			SWTLOCATION* rect = &(group_data->groups[index].location);
			int text_w = rect->maxx - rect->minx + 1;
			int text_h = rect->maxy - rect->miny + 1;
			int area = group_data->groups[index].area;
			float mean = sumx[index] / area;
			float var = sumxx[index] / area - mean * mean;


			group_data->groups[index].label *= -1;

			if(area < text_min_size)
				continue;//size

			if(min(text_w, text_h) * text_max_aspect < max(text_w, text_h))
				continue;//aspect ratio
			
			if(text_h < text_min_h || text_h > text_max_h)
				continue; //font height

			if(var > sw_max_var)
				continue; //variance


			sw_list.clear();
			for(row = rect->miny; row <= rect->maxy; row++)
			{
				for(col = rect->minx; col <= rect->maxx; col++)
				{

					if(label_data.at<int>(row,col) != index + 1)
					{
						continue;
					}
					sw_list.push_back(swt_data->sw(row,col));
				}
			}

			sort(sw_list.begin(), sw_list.end(), greater<float>());


			float sw = sw_list[sw_list.size() / 2]; //用中位数表示改组的SW

			if(max(text_h, text_w) / sw > text_max_size2sw)
				continue;


			group_data->groups[index].sw = sw;
			group_data->groups[index].label *= -1;

		}

	}


	delete[] sumx;
	delete[] sumxx;

	return 0;

}







CSWT* InitSWData(cv::Mat canny_data, cv::Mat gray_data)
{//确定Stroke width

	CSWT* swt_data = new CSWT(cv::Size(canny_data.cols, canny_data.rows));

	int row,col;

	int width = canny_data.cols, height = canny_data.rows;

	cv::Mat smooth_data, gx_data, gy_data;
	cv::GaussianBlur(gray_data, gray_data, cv::Size(3, 3), 1, 1);
	cv::Sobel(gray_data, gx_data, CV_32FC1, 1, 0);
	cv::Sobel(gray_data, gy_data, CV_32FC1, 0, 1);


	for(row = 2; row + 2 < height; row++)
	{
		for(col = 2; col + 2 < width; col++)
		{
			double dx = 0, dy = 0;
			double d = 0;

			if (canny_data.at<uchar>(row, col) == 0) continue;

			dx = gx_data.at<float>(row, col);
			dy = gy_data.at<float>(row, col);

			d = sqrt(dx * dx + dy * dy);
			if(d <= FLT_EPSILON)
			{
				dx = 0;
				dy = 0;
			}
			else
			{
				dx /= d;
				dy /= d;
			}
			swt_data->edge(row, col) = 1;
			swt_data->dx(row, col) = dx;
			swt_data->dy(row, col) = dy;
		}
	}




	return swt_data;
}


//work for white font
int CalcSW(cv::Mat canny_data, CSWT* swt_data, int nMaxSW)
{
	int row, col;
	int nWidth = canny_data.cols;
	int nHeight = canny_data.rows;


	for (row = 0; row < nHeight; row++)
	{
		for (col = 0; col < nWidth; col++)
		{
			swt_data->sw(row, col) = FLT_MAX;
		}
	}

	//1--first round of SWT
	for (row = 0; row + 0 < nHeight; row++)
	{
		for (col = 0; col + 0 < nWidth; col++)
		{

			int nCx = col;
			int nCy = row;

			int nSW;
			double dx, dy;

			int bRay = 0;

			double curdeg = 0;
			int swlen = nMaxSW;


			if(swt_data->edge(row,col) == 0)
				continue;

			dx = swt_data->dx(row, col);
			dy = swt_data->dy(row, col);
			curdeg = atan2(dy, dx);
			curdeg = curdeg < 0 ? __max(curdeg + 2 * SWTPI, SWTPI) : curdeg;


			for (nSW = 1; nSW < swlen && bRay == 0; nSW++)
			{
				int x = col + (int)(nSW * dx);
				int y = row + (int)(nSW * dy);

				if (x < 0 || y < 0 || x >= nWidth || y >= nHeight)
					break;

				if(swt_data->edge(y,x))
				{
					double deg = atan2(swt_data->dy(y,x), swt_data->dx(y,x));
					deg = deg < 0 ? __max(deg + 2 * SWTPI, SWTPI) : deg;

					//-degree
					deg = deg > SWTPI ? deg - SWTPI : deg + SWTPI;

					if (abs(deg - curdeg) * 6 < SWTPI)
					{
						bRay = 1;
					}
				}
			}


			swt_data->ray(row, col) = bRay;
			swt_data->raylen(row, col) = nSW;

			if (bRay)
			{
				//find one stroke and update the stroke width for each pixel inside
				int nStrokeWidth = nSW;
				while (nSW >= 0)
				{
					int x = col + (int)(dx * nSW);
					int y = row + (int)(dy * nSW);

					double v = swt_data->sw(y, x);
					v = __min(v, nStrokeWidth);
					swt_data->sw(y, x) = v;
					nSW--;
				}
			}



		}
	}

	//2--second round
	{
		vector<double> vSW;
		for (row = 0; row < nHeight; row++)
		{
			for (col = 0; col < nWidth; col++)
			{
				double fMidV;
				int nLen, nMaxLen;
				if(swt_data->ray(row,col) == 0)
				{
					continue;
				}
				nMaxLen = swt_data->raylen(row, col);
				vSW.clear();
				for (nLen = 0; nLen < nMaxLen; nLen++)
				{
					vSW.push_back( swt_data->sw(row,col)  );
				}


				sort(vSW.begin(), vSW.end(), greater<double>());

				fMidV = vSW[vSW.size() / 2];

				for (nLen = 0; nLen < nMaxLen; nLen++)
				{
					double v = swt_data->sw(row, col);
					swt_data->sw(row,col) = min(v, fMidV);
				}
			}
		}

	}


	//using -1 to replace FLT_MAX
	for (row = 0; row < nHeight; row++)
	{
		for (col = 0; col < nWidth; col++)
		{
			double v = swt_data->sw(row, col);
			if (v * 2 > FLT_MAX - 3)
			{
				swt_data->sw(row, col) = -1;
			}
		}
	}


	return 0;
}


int cvtConnect(cv::Mat mask_data)
{
	int row, col;

	cv::Mat backup_data = mask_data.clone();

	for (row = 1; row + 1 < mask_data.rows; row++)
	{
		for (col = 1; col + 1 < mask_data.cols; col++)
		{
			int x, y;

			if (backup_data.at<uchar>(row, col) == 0) continue;


			x = col - 1;
			y = row - 1;
			unsigned char val = backup_data.at<uchar>(y, x);
			mask_data.at<uchar>(row-1, col) = std::max<uchar>(val, mask_data.at<uchar>(row-1, col));
			mask_data.at<uchar>(row, col - 1) = std::max<uchar>(val, mask_data.at<uchar>(row, col-1));

			x = col + 1;
			y = row - 1;
			val = backup_data.at<uchar>(y, x);
			mask_data.at<uchar>(row-1, col) = std::max<uchar>(val, mask_data.at<uchar>(row-1, col));
			mask_data.at<uchar>(row, col + 1) = std::max<uchar>(val, mask_data.at<uchar>(row, col+1));

			x = col + 1;
			y = row + 1;
			val = backup_data.at<uchar>(y, x);
			mask_data.at<uchar>(row+1, col) = std::max<uchar>(val, mask_data.at<uchar>(row+1, col));
			mask_data.at<uchar>(row, col + 1) = std::max<uchar>(val, mask_data.at<uchar>(row, col+1));



			x = col - 1;
			y = row + 1;
			val = backup_data.at<uchar>(y, x);
			mask_data.at<uchar>(row+1, col) = std::max<uchar>(val, mask_data.at<uchar>(row+1, col));
			mask_data.at<uchar>(row, col - 1) = std::max<uchar>(val, mask_data.at<uchar>(row, col-1));
		}
	}
	return 0;
}

int SWTTextDetection(cv::Mat image, std::vector<cv::Rect>& texts)
{
	int nMaxSW = 15;
	double fMinRatio = 0;
	double fMaxRatio = 2;

	cv::Mat canny_data;

	cv::Mat gray_data;
	if (image.channels() == 1)
		gray_data = image.clone();
	else
		cv::cvtColor(image, gray_data, cv::COLOR_BGR2GRAY);

	cv::Canny(gray_data, canny_data, 150, 200, 3);


	cvtConnect(canny_data);



	CSWT* swt_data = InitSWData(canny_data, gray_data);


	CalcSW(canny_data, swt_data, nMaxSW);

	CSWTGROUP group_data;
	GroupSW(swt_data,  fMinRatio, fMaxRatio,&group_data);


	TextFilter(swt_data,&group_data);

	for (int k = 0; k < group_data.groups.size(); k++)
	{
		cv::Rect rect;
		SWTLOCATION r0 = group_data.groups[k].location;
		rect.x = r0.minx;
		rect.y = r0.miny;
		rect.width = r0.maxx - r0.minx + 1;
		rect.height = r0.maxy - r0.miny + 1;
		texts.push_back(rect);
	}



	delete swt_data;
	return 0;
}