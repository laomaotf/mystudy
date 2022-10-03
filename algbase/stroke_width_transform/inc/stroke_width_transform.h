#pragma one
#include <vector>
#include "opencv2/opencv.hpp"


/*
Detecting Text in Natural Scenes with Stroke Width Transform
Boris Epshtein,Eyal Ofek,Yonatan Wexler

1. SWTһ��ֻ�ܼ��һ����ɫ������
   �ݶȷ��򣬿��Թ涨Ϊָ��ҶȽϸߵ����򣨵�Ȼ�෴Ҳ���ԣ�����ʱ�������Ҷȼ����ڱ����������ݶȷ�������
   ����������ȷ�ģ���֮�������ݶȷ�������ҵ�������һ���ַ�������

2. SWT�����ݶȷ���ȷ��stroke widthʱ����Ҫcanny 4��ͨ��Ե���������׵����Ҳ�����Ӧ��������

3. SWT������ͼƬЧ���ȽϺã�ģ����ͼƬ��ȡ�߽粻׼ȷ��Ч�����

*/

int SWTTextDetection(cv::Mat image, std::vector<cv::Rect>& texts);


