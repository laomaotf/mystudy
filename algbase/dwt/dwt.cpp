#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "dwt.h"

namespace dwt
{
	/*
	函数名: step_forward_d2
	功能: 一级小波变换(D2)
	参数: pImage          		    原始图像
		  nWidthEven                   需要变换的图像宽度
		  nHeightEven                   需要变换的图像高度
		  nWidth                   图像宽度
		  nHeight                  图像高度
		  pDWTData[out]             变换结果 (nHeight, nWidth)
	返回值:  0                      错误
			 1                      正常
	*/
	int step_forward_d2(float* pImage, int nWidthEven, int nHeightEven, int nWidth, int nHeight, float* pDWTData)
	{
		float lower_filter[2] = { 0.5f, 0.5f };
		float higher_filter[2] = { 0.5f, -0.5f };

		//parameters check
		if (pImage == NULL || pDWTData == NULL)
			return 0;
		if (nWidthEven < 1 || nHeightEven < 1 || nWidth < 1 || nHeight < 1)
			return 0;
		if (nWidthEven > nWidth || nHeightEven > nHeight)
			return 0;
		if (nWidthEven % 2 != 0 || nHeightEven % 2 != 0) //even is just OK
			return 0;

		int iRow, iCol;
		float* pBuf = NULL;
		//horizontal transform
		pBuf = (float*)malloc(sizeof(float*) * nWidthEven);
		for (iRow = 0; iRow < nHeightEven; iRow++)
		{
			for (iCol = 0; iCol < nWidthEven / 2; iCol++)
			{
				pBuf[iCol] = lower_filter[0] * pImage[iRow * nWidth + 2 * iCol]
					+ lower_filter[1] * pImage[iRow * nWidth + 2 * iCol + 1];

				pBuf[iCol + nWidthEven / 2] = higher_filter[0] * pImage[iRow * nWidth + 2 * iCol]
					+ higher_filter[1] * pImage[iRow * nWidth + 2 * iCol + 1];
			}
			memcpy(pDWTData + nWidth * iRow, pBuf, sizeof(float) * nWidthEven);
		}

		//vertical transform
		pBuf = (float*)malloc(sizeof(float*) * nHeightEven);
		for (iCol = 0; iCol < nWidthEven; iCol++)
		{
			for (iRow = 0; iRow < nHeightEven / 2; iRow++)
			{
				pBuf[iRow] = lower_filter[0] * pDWTData[2 * iRow * nWidth + iCol]
					+ lower_filter[1] * pDWTData[(2 * iRow + 1) * nWidth + iCol];
				pBuf[iRow + nHeightEven / 2] = higher_filter[0] * pDWTData[2 * iRow * nWidth + iCol]
					+ higher_filter[1] * pDWTData[(2 * iRow + 1) * nWidth + iCol];
			}
			for (iRow = 0; iRow < nHeightEven; iRow++)
			{
				pDWTData[iRow * nWidth + iCol] = pBuf[iRow];
			}
		}
		free(pBuf);
		return 1;
	}


	/*
	函数名: step_backward_d2
	功能: 一级逆小波变换(D2)
	参数: pDWTData                  原始数据
		  nWidthEven                   需要变换的图像宽度
		  nHeightEven                   需要变换的图像高度
		  nWidth                   图像宽度
		  nHeight                  图像高度
		  pImage[out]             变换结果 (nHeight, nWidth)
	返回值:  0                      错误
			 1                      正常
	*/
	int step_backward_d2(float* pDWTData, int nWidthEven, int nHeightEven, int nWidth, int nHeight, float* pImage)
	{
		if (pDWTData == NULL || pImage == NULL)
			return 0;

		size_t float_byte = sizeof(float);
		int iRow, iCol;
		float* pBuf = NULL;

		//vertical transform
		pBuf = (float*)malloc(nHeightEven * float_byte);
		for (iCol = 0; iCol < nWidthEven; iCol++)
		{
			for (iRow = 0; iRow < nHeightEven / 2; iRow++)
			{
				pBuf[2 * iRow] = pDWTData[iRow * nWidth + iCol] + pDWTData[(iRow + nHeightEven / 2) * nWidth + iCol];
				pBuf[2 * iRow + 1] = pDWTData[iRow * nWidth + iCol] - pDWTData[(iRow + nHeightEven / 2) * nWidth + iCol];
			}
			for (iRow = 0; iRow < nHeightEven; iRow++)
				pImage[iRow * nWidth + iCol] = pBuf[iRow];
		}
		free(pBuf);

		//horizontal transform
		pBuf = (float*)malloc(float_byte * nWidthEven);
		for (iRow = 0; iRow < nHeightEven; iRow++)
		{
			for (iCol = 0; iCol < nWidthEven / 2; iCol++)
			{
				pBuf[2 * iCol] = pImage[iRow * nWidth + iCol] + pImage[iRow * nWidth + iCol + nWidthEven / 2];
				pBuf[2 * iCol + 1] = pImage[iRow * nWidth + iCol] - pImage[iRow * nWidth + iCol + nWidthEven / 2];
			}
			memcpy(pImage + iRow * nWidth, pBuf, nWidthEven * float_byte); //take care of the addition of pointer, it is type-dependent!
		}
		free(pBuf);
		return 1;
	}


	/*
	函数名: step_forward
	功能: 一级小波变换
	参数: pImage          		    原始图像
		  nWidthEven                   需要变换的图像宽度
		  nHeightEven                   需要变换的图像高度
		  nWidth                   图像宽度
		  nHeight                  图像高度
		  nType                    小波类型(0-D2 1-D4 2-D6 3-D8)
		  pDWTData[out]             变换结果 (nHeight, nWidth)
	返回值:  0                      错误
			 1                      正常
	*/
	int step_forward(float* pImage, int nWidthEven, int nHeightEven, int nWidth, int nHeight, int nType, float* pDWTData)
	{
		float lFilter_D4[4] = { 0.482962913f / 1.414f,0.836516303 / 1.414,0.224143868 / 1.414,-0.129409522 / 1.414 },
			hFilter_D4[4] = { 0.129409522f / 1.414f,0.224143868 / 1.414,-0.836516303 / 1.414,0.482962913 / 1.414 };
		float lFilter_D6[6] = { 0.332670552950f / 1.414f,0.806891509311 / 1.414,0.459877502118 / 1.414,-0.135011020010 / 1.414,-0.085441273882 / 1.414,0.035226291882 / 1.414 },
			hFilter_D6[6] = { 0.035226291882f / 1.414f,0.085441273882 / 1.414,-0.135011020010 / 1.414,-0.459877502118 / 1.414,0.806891509311 / 1.414,-0.332670552950 / 1.414 };
		float lFilter_D8[8] = { 0.230377813309f / 1.414f,0.714846570553 / 1.414,0.630880767930 / 1.414,-0.027983769417 / 1.414,-0.187034811718 / 1.414,0.030841381836 / 1.414,0.032883011667 / 1.414,-0.010597401785 / 1.41 },
			hFilter_D8[8] = { -0.010597401785f / 1.414f,-0.032883011667 / 1.414,0.030841381 / 1.414,0.187034811719 / 1.414,-0.027983769417 / 1.414,-0.630880767930 / 1.414,0.714846570553 / 1.414,-0.230377813309 / 1.414 };

		int FilterLen = 0;
		float* lFilter, * hFilter;
		switch (nType) {
		case 0:
			step_forward_d2(pImage, nWidthEven, nHeightEven, nWidth, nHeight, pDWTData);
			return 1;
		case 1:
			FilterLen = 4;
			lFilter = lFilter_D4;
			hFilter = hFilter_D4;
			break;
		case 2:
			FilterLen = 6;
			lFilter = lFilter_D6;
			hFilter = hFilter_D6;
			break;
		case 3:
			FilterLen = 8;
			lFilter = lFilter_D8;
			hFilter = hFilter_D8;
			break;
		default:
			return 0;
		}

		size_t float_byte = sizeof(float);
		float* pBuf = NULL;
		int iRow, iCol, index;
		int temp;

		//horizontal transform
		pBuf = (float*)malloc(float_byte * nWidthEven);
		for (iRow = 0; iRow < nHeightEven; iRow++)
		{
			memset(pBuf, 0, float_byte * nWidthEven);
			for (iCol = 0; iCol < nWidthEven / 2; iCol++)
			{
				for (index = 0; index < FilterLen; index++)
				{
					temp = 2 * iCol + index;
					temp = temp % nWidthEven; //周期延拓
					pBuf[iCol] += pImage[iRow * nWidth + temp] * lFilter[index];
					pBuf[iCol + nWidthEven / 2] += pImage[iRow * nWidth + temp] * hFilter[index];
				}
			}
			memcpy(pDWTData + nWidth * iRow, pBuf, float_byte * nWidthEven);
		}
		free(pBuf);

		//vertical transform
		pBuf = (float*)malloc(float_byte * nHeightEven);
		for (iCol = 0; iCol < nWidthEven; iCol++)
		{
			memset(pBuf, 0, float_byte * nHeightEven);
			for (iRow = 0; iRow < nHeightEven / 2; iRow++)
			{
				for (index = 0; index < FilterLen; index++)
				{
					temp = 2 * iRow + index;
					temp = temp % nHeightEven;//周期延拓
					pBuf[iRow] += lFilter[index] * pDWTData[temp * nWidth + iCol];
					pBuf[iRow + nHeightEven / 2] += hFilter[index] * pDWTData[temp * nWidth + iCol];
				}
			}
			for (iRow = 0; iRow < nHeightEven; iRow++)
				pDWTData[iRow * nWidth + iCol] = pBuf[iRow];
		}
		free(pBuf);
		return 1;
	}


	/*
	函数名: step_backward
	功能: 一级小波逆变换
	参数: pDWTData                  原始数据
		  nWidthEven                   需要变换的数据宽度
		  nHeightEven                   需要变换的数据高度
		  nWidth                   数据宽度
		  nHeight                  数据高度
		  nType                    小波类型(0-D2 1-D4 2-D6 3-D8)
		  pImage          		    还原图像
	返回值:  0                      错误
			 1                      正常
	*/
	int step_backward(float* pDWTData, int nWidthEven, int nHeightEven, int nWidth, int nHeight, int nType, float* pImage)
	{
		float  lFilter_D4[4] = { 0.482962913 / 1.414,0.836516303 / 1.414,0.224143868 / 1.414,-0.129409522 / 1.414 },
			hFilter_D4[4] = { 0.129409522 / 1.414,0.224143868 / 1.414,-0.836516303 / 1.414,0.482962913 / 1.414 };
		float  lFilter_D6[6] = { 0.332670552950 / 1.414,0.806891509311 / 1.414,0.459877502118 / 1.414,-0.135011020010 / 1.414,-0.085441273882 / 1.414,0.035226291882 / 1.414 },
			hFilter_D6[6] = { 0.035226291882 / 1.414,0.085441273882 / 1.414,-0.135011020010 / 1.414,-0.459877502118 / 1.414,0.806891509311 / 1.414,-0.332670552950 / 1.414 };
		float  lFilter_D8[8] = { 0.230377813309 / 1.414,0.714846570553 / 1.414,0.630880767930 / 1.414,-0.027983769417 / 1.414,-0.187034811718 / 1.414,0.030841381836 / 1.414,0.032883011667 / 1.414,-0.010597401785 / 1.41 },
			hFilter_D8[8] = { -0.010597401785 / 1.414,-0.032883011667 / 1.414,0.030841381 / 1.414,0.187034811719 / 1.414,-0.027983769417 / 1.414,-0.630880767930 / 1.414,0.714846570553 / 1.414,-0.230377813309 / 1.414 };

		int dwFilterLen = 0;
		float* lFilter, * hFilter;
		switch (nType)
		{
		case 0:
			step_backward_d2(pDWTData, nWidthEven, nHeightEven, nWidth, nHeight, pImage);
			return 1;
		case 1:
			dwFilterLen = 4;
			lFilter = lFilter_D4;
			hFilter = hFilter_D4;
			break;
		case 2:
			dwFilterLen = 6;
			lFilter = lFilter_D4;
			hFilter = hFilter_D6;
			break;
		case 3:
			dwFilterLen = 8;
			lFilter = lFilter_D8;
			hFilter = hFilter_D8;
			break;
		default:
			return 0;
		}

		size_t float_byte = sizeof(float);
		int iRow, iCol, index;
		int temp;
		float* pBuf = NULL;

		//vertical transform
		pBuf = (float*)malloc(float_byte * nHeightEven);
		for (iCol = 0; iCol < nWidthEven; iCol++)
		{
			memset(pBuf, 0, float_byte * nHeightEven);
			for (iRow = 0; iRow < nHeightEven; iRow++)
			{
				for (index = 0; index < dwFilterLen; index++)
				{
					temp = iRow - index + nHeightEven; //周期延拓
					temp = temp % nHeightEven;
					if (temp % 2 == 0)
						pBuf[iRow] += lFilter[index] * pDWTData[(temp >> 1) * nWidth + iCol]
						+ hFilter[index] * pDWTData[((temp >> 1) + nHeightEven / 2) * nWidth + iCol];
				}
			}

			for (iRow = 0; iRow < nHeightEven; iRow++)
				pImage[iRow * nWidth + iCol] = 2 * pBuf[iRow];
		}
		free(pBuf);

		//horizontal transform
		pBuf = (float*)malloc(nWidthEven * float_byte);
		for (iRow = 0; iRow < nHeightEven; iRow++)
		{
			memset(pBuf, 0, nWidthEven * float_byte);
			for (iCol = 0; iCol < nWidthEven; iCol++)
			{
				for (index = 0; index < dwFilterLen; index++)
				{
					temp = iCol - index + nWidthEven; //周期延拓
					temp = temp % nWidthEven;
					if (temp % 2 == 0)
						pBuf[iCol] += lFilter[index] * pImage[iRow * nWidth + (temp >> 1)]
						+ hFilter[index] * pImage[iRow * nWidth + (temp >> 1) + nWidthEven / 2];
				}
			}
			//		memcpy(pImage+nWidth*iRow,pBuf,nWidthEven*float_byte);
			for (iCol = 0; iCol < nWidthEven; iCol++)
				pImage[iRow * nWidth + iCol] = 2 * pBuf[iCol]; //why 2 is multiplied???
		}
		free(pBuf);
		return 1;
	}



	int backward(float* pDWTData, int nWidth, int nHeight, int nDepth, int nType, float* pImage)
	{
		if (pDWTData == NULL || pImage == NULL)
			return 0;

		//calculate the size of the top layer 
		int nWidthEven = nWidth >> (nDepth - 1);
		int nHeightEven = nHeight >> (nDepth - 1);

		memcpy(pImage, pDWTData, sizeof(float) * nWidth * nHeight);
		nDepth++;
		while (nDepth > 1)
		{
			step_backward(pDWTData, nWidthEven, nHeightEven, nWidth, nHeight, nType, pImage);
			pDWTData = pImage;
			nWidthEven *= 2;
			nHeightEven *= 2;

			nDepth--;
		}
		return 1;
	}


	int forward(float* pImage, int nWidth, int nHeight, int nDepth, int nType, float* pDWTData)
	{
		if (pImage == NULL || pDWTData == NULL)
			return 0;
		if (nDepth < 1 || nHeight < 1 || nWidth < 1)
			return 0;

		int nWidthEven = nWidth;
		int nHeightEven = nHeight;

		nDepth++;
		while (nDepth > 1)
		{
			step_forward(pImage, nWidthEven, nHeightEven, nWidth, nHeight, nType, pDWTData);
			pImage = pDWTData;
			nWidthEven = nWidthEven >> 1;
			nHeightEven = nHeightEven >> 1;
			nDepth--;
		}
		return 1;
	}

}
