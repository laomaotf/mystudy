#pragma once

namespace dwt
{
	/*
	������: forward
	����: �༶С�����任
	����: pImage          		    ԭʼͼ��
		  nWidth                   ͼ����
		  nHeight                  ͼ��߶�
		  nDepth                   �任����
		  nType                    С������(0-D2 1-D4 2-D6 3-D8)
		  pDWTData                  �任���(out)
	����ֵ:  0                      ����
			 1                      ����
	*/
	int forward(float* pImage, int nWidth, int nHeight, int nDepth, int nType, float* pDWTData);

	/*
	������: backward
	����: �༶С����任
	����: pDWTData                  ԭʼ����
		  nWidth                   ͼ����
		  nHeight                  ͼ��߶�
		  nDepth                   �任����
		  nType                    С������(0-D2 1-D4 2-D6 3-D8)
		  pImage          		    ��ԭͼ��(out)
	����ֵ:  0                      ����
			 1                      ����
	*/
	int backward(float* pDWTData, int nWidth, int nHeight, int nDepth, int nType, float* pImage);

}
