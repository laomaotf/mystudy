#pragma once

namespace dwt
{
	/*
	函数名: forward
	功能: 多级小波正变换
	参数: pImage          		    原始图像
		  nWidth                   图像宽度
		  nHeight                  图像高度
		  nDepth                   变换级数
		  nType                    小波类型(0-D2 1-D4 2-D6 3-D8)
		  pDWTData                  变换结果(out)
	返回值:  0                      错误
			 1                      正常
	*/
	int forward(float* pImage, int nWidth, int nHeight, int nDepth, int nType, float* pDWTData);

	/*
	函数名: backward
	功能: 多级小波逆变换
	参数: pDWTData                  原始数据
		  nWidth                   图像宽度
		  nHeight                  图像高度
		  nDepth                   变换级数
		  nType                    小波类型(0-D2 1-D4 2-D6 3-D8)
		  pImage          		    还原图像(out)
	返回值:  0                      错误
			 1                      正常
	*/
	int backward(float* pDWTData, int nWidth, int nHeight, int nDepth, int nType, float* pImage);

}
