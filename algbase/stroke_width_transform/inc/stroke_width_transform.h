#pragma one
#include <vector>
#include "opencv2/opencv.hpp"


/*
Detecting Text in Natural Scenes with Stroke Width Transform
Boris Epshtein,Eyal Ofek,Yonatan Wexler

1. SWT一次只能检测一种颜色的字体
   梯度方向，可以规定为指向灰度较高的区域（当然相反也可以），此时如果字体灰度级高于背景，则沿梯度方向搜索
   轮廓点是正确的；反之，则沿梯度方向可能找到的是另一个字符的轮廓

2. SWT利用梯度方向确定stroke width时，需要canny 4联通边缘，否则容易导致找不到对应的轮廓点

3. SWT对清晰图片效果比较好，模糊的图片提取边界不准确，效果变差

*/

int SWTTextDetection(cv::Mat image, std::vector<cv::Rect>& texts);


