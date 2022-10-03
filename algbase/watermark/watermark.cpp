#include "watermark.h"
namespace watermark
{

    class FFT2D
    {
    public:
        FFT2D() {}
        ~FFT2D() {}
    public:

        static int forward(const cv::Mat& src, cv::Mat& Fourier)
        {
            int channels = src.channels();
            if (channels > 2) return -1;

            if (channels == 1)
            {
                cv::Mat src_dbl;
                if (src.type() == CV_8UC1)
                    src.convertTo(src_dbl, CV_64F, 1.0 / 255);
                else if (src.type() == CV_64FC1)
                    src_dbl = src;
                else
                    return -1;
                cv::Mat planes[] = { src_dbl, cv::Mat::zeros(src.size(),CV_64F) };
                merge(planes, 2, Fourier);
                cv::dft(Fourier, Fourier);
            }
            else
            {
                //cv::Mat src_dbl;
                //if (src.type() != CV_64FC2)
                //    src.convertTo(src_dbl, CV_64FC2);
                //else
                //    src_dbl = src;

                //cv::Mat tmp;
                //cv::dft(src_dbl, tmp);
                //std::vector<cv::Mat> planes;
                //cv::split(tmp, planes);
                //cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
                //Fourier = planes[0];
                return -1;
            }

            return 0;
        }

        static int backward(const cv::Mat& src, cv::Mat& Fourier)
        {
            int channels = src.channels();
            if (channels > 2) return -1;

            if (channels == 1)
            {
                //cv::Mat src_dbl;
                //if (src.type() != CV_64F)
                //    src.convertTo(src_dbl, CV_64F);
                //else
                //    src_dbl = src;

                //cv::Mat planes[] = { src_dbl, cv::Mat::zeros(src_dbl.size(),CV_64F) };
                //merge(planes, 2, Fourier);
                //dft(Fourier, Fourier, cv::DFT_INVERSE + cv::DFT_SCALE, 0);


                //std::vector<cv::Mat> planes_time;
                //cv::split(Fourier, planes_time);
                //cv::magnitude(planes_time[0], planes_time[1], planes_time[0]); //将复数转化为幅值
                //Fourier = planes_time[0];
                return -1;
            }
            else // 7<mat_type<15
            {
                cv::Mat src_dbl;
                if (src.type() != CV_64FC2)
                    src.convertTo(src_dbl, CV_64FC2);
                else
                    src_dbl = src;

                cv::Mat tmp;
                cv::dft(src_dbl, tmp, cv::DFT_INVERSE + cv::DFT_SCALE, 0);
                std::vector<cv::Mat> planes;
                cv::split(tmp, planes);
                cv::magnitude(planes[0], planes[1], planes[0]); //将复数转化为幅值
                Fourier = planes[0];
            }
            return 0;
        }
    };

    class FFT2DRGB
    {
    public:
        FFT2DRGB() {}
        ~FFT2DRGB() {}
    public:
        static int forward(cv::Mat& input, cv::Mat& output)
        {
            std::vector<cv::Mat> channels_time, channels_fft;
            cv::split(input, channels_time);
            for (auto one : channels_time)
            {
                cv::Mat one_fft;
                int ret = FFT2D::forward(one, one_fft);
                if (ret != 0) return -1;
                channels_fft.push_back(one_fft);
            }
            cv::merge(channels_fft, output);
            return 0;
        }

        static int backward(cv::Mat& input, cv::Mat& output)
        {
            std::vector<cv::Mat> channels_fft, channels_time;
            cv::split(input, channels_fft);
            if (channels_fft.size() % 2 != 0) return -1;
            for (int k = 0; k < channels_fft.size(); k += 2)
            {
                cv::Mat one_time, one_fft;
                std::vector<cv::Mat> tmp_fft = { channels_fft[k], channels_fft[k + 1] };
                cv::merge(tmp_fft, one_fft);
                FFT2D::backward(one_fft, one_time);
                channels_time.push_back(one_time);
            }
            cv::merge(channels_time, output);
            return 0;
        }
    };

    int find_resize_ratio(INPUT& input)
    {
        int scale = 1;
        do
        {
            if (input.watermark.cols / scale < input.image.cols / 2 && input.watermark.rows / scale < input.image.rows / 2)
                break;
            scale += 1;
        } while (1);
        return scale;
    }


    int WatermarkEncodingMethod(cv::Size size, std::vector<int>& xmap, std::vector<int>& ymap)
    {
        if (1)
        {
            for (int x = 0; x < size.width; x++)
            {
                xmap.push_back(x);
            }
            for (int y = 0; y < size.height; y++)
            {
                ymap.push_back(y);
            }
            //随机打乱watermark,有助于提高水印稳定性
            std::random_shuffle(xmap.begin(), xmap.end());
            std::random_shuffle(ymap.begin(), ymap.end());
        }
        else
        {
            for (int x = 0; x < size.width; x++)
            {
                xmap.push_back(x);
            }
            for (int y = 0; y < size.height; y++)
            {
                ymap.push_back(y);
            }
        }
        return 0;
    }

    int Encode(INPUT& input, OUTPUT& output)
    {

        if (input.alpha < 3)
        {
            std::cout << "WARNING: alpha should be larger enough. and suggest minimum alpha is 3" << std::endl;
        }

 /*       if (input.image.channels() != 3)
        {
            std::cout << "ERROR: image channels expected to 3 and "<<input.image.channels()<<" inputed" << std::endl;
            return -1;
        }*/

        //////////////////////////
        //fft for image
        cv::Mat image_time;
        cv::Mat image_fft;
       // input.image.convertTo(image_time, CV_64FC1, 1 / 255.0);
        //int ret = fft2(image_time,image_fft);
        int ret = FFT2DRGB::forward(input.image, image_fft);


        /////////////////////////////////
        //resize watermark to be smaller than image
        cv::Mat watermark;
        int scale = find_resize_ratio(input);
        cv::Size dsize = cv::Size(input.watermark.cols / scale, input.watermark.rows / scale);
        if (dsize.width < 1 || dsize.height < 1) return -1;
        cv::resize(input.watermark, watermark, dsize);
        if (watermark.channels() != 1) cv::cvtColor(watermark, watermark, cv::COLOR_BGR2GRAY);
        cv::Mat watermark_float;
        watermark.convertTo(watermark_float, CV_64FC1, 1 / 255.0);

        ////////////////////////////////////////////////////////////
        //创建一个和image等大的图，把watermark对称加入图片(水印图片要关于中心点对成)
        //单通道，后续同时作用到虚部和实部
        cv::Mat watermark_padding = cv::Mat::zeros(image_fft.size(), CV_64FC1);
        for (int y = 0; y < watermark_float.rows; y++)
        {
            for (int x = 0; x < watermark_float.cols; x++)
            {
                double val = watermark_float.at<double>(y,x);
				watermark_padding.at<double>(y, x) = val;
				watermark_padding.at<double>(watermark_padding.rows - 1 - y, (watermark_padding.cols - 1 - x)) = val;
            }
        }
        //encoding+blend(利用watermark padding后随机编码，加入image fft)
        std::vector<int> xmap, ymap;
        WatermarkEncodingMethod(image_fft.size(), xmap, ymap);
        cv::Mat image_fft_blend = cv::Mat::zeros(image_fft.size(), CV_64FC(image_fft.channels()));
        for (int y = 0; y < watermark_padding.rows; y++)
        {
            for (int x = 0; x < watermark_padding.cols; x++)
            {
                double val = watermark_padding.at<double>(ymap[y], xmap[x]);
                for (int c = 0; c < image_fft.channels(); c++)
                {
                    image_fft_blend.at<double>(y, x * image_fft.channels() + c) = image_fft.at<double>(y, x * image_fft.channels() + c) + val * input.alpha;
                }
            }
        }


        ///////////////////////////////////////////////////////////////////
        //ifft2
        FFT2DRGB::backward(image_fft_blend, image_time);
        image_time.convertTo(output.image, CV_8U, 255.0);

        ///////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////
        ////extract watermark
        cv::Mat image_rebuild_fft; //和image_fft_blend 对应
        ret = FFT2DRGB::forward(output.image, image_rebuild_fft);
        cv::Mat residual = (image_rebuild_fft - image_fft); //和乱序的watermark_padding对应
        cv::Mat watermark_rebuild = cv::Mat::zeros(residual.size(), CV_64FC1);
        //按照编码规则恢复watermark
        for (int y = 0; y < residual.rows; y++)
        {
            for (int x = 0; x < residual.cols; x++)
            {
                watermark_rebuild.at<double>(ymap[y], xmap[x]) = residual.at<double>(y, x * image_rebuild_fft.channels() + 0) / input.alpha;
            }
        }
        watermark_rebuild.convertTo(output.watermak, CV_8U, 255.0);
        return 0;
    }



};