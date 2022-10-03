#include <iostream>
#include <fstream>
#include "beam_decode.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
using namespace logging::trivial;
bool SaveData(std::string path, cv::Mat& mat, beam_decoder::OUTPUT& output)
{
	
	rapidjson::StringBuffer buffer;
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
	writer.StartObject();



	//OUTPUT
	writer.Key("output_sentences");
	writer.StartArray();
	for (auto w : output.beam) 
	{
		writer.Int(w);
	}
	writer.EndArray();

	writer.Key("output_probs");
	writer.Double(output.prob);

	//MAT(rows,cols,data)
	writer.Key("input");
	writer.StartObject();
	writer.Key("rows"), writer.Int(mat.cols);
	writer.Key("cols"), writer.Int(mat.rows);
	writer.Key("data");
	writer.StartArray();
	for (int row = 0; row < mat.rows; row++)
	{
		for (int col = 0; col < mat.cols; col++)
		{
			writer.Double(mat.at<float>(row, col));
		}
	}
	writer.EndArray();
	writer.EndObject();

	writer.EndObject();
	//WRITE DISK
	std::ofstream fd(path);
	fd << buffer.GetString();
	fd.close();
	return true;
}

void setup_boost_logging()
{
	logging::add_file_log(
		keywords::file_name = "beam_decode_%N.log",                                        /*< file name pattern >*/
		keywords::rotation_size = 10 * 1024 * 1024,                                   /*< rotate files every 10 MiB... >*/
		keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/
		keywords::format = "[%TimeStamp%] [%LineID%]: %Message%"
	);
	logging::core::get()->set_filter(logging::trivial::severity >= boost::log::trivial::info);
	logging::add_common_attributes();
}

int main(int argc, char* argv[])
{

	bool load_data_from_disk = false;
	setup_boost_logging();

	const int T = 10, S = 5;
	cv::Mat prob_mat(cv::Size(S, T), CV_32FC1);
	cv::randu(prob_mat, 0, 100);
	for (int t = 0; t < T; t++)
	{
		double m0, m1;
		cv::minMaxLoc(prob_mat.row(t), &m0, &m1);
		for (int s = 0; s < S; s++) prob_mat.at<float>(t, s) /= m1;
	}

	if(load_data_from_disk)
	{
		cv::FileStorage fs("../python/data.yml", cv::FileStorage::WRITE);
		fs << "prob" << prob_mat;
		fs.release();
	}
	else
	{
		cv::FileStorage fs("../python/data.yml", cv::FileStorage::READ);
		fs["prob"] >> prob_mat;
		fs.release();

	}


	beam_decoder::INPUT input;
	beam_decoder::OUTPUT output;
	input.beam_size = 2;
	input.probs = prob_mat;


	beam_decoder::FirstPassLM(input, output, 0);

	{
		std::cout << output.prob << " : ";
		for (auto w : output.beam)
		{
			std::cout << w << ",";
		}
		std::cout << std::endl;
	}

	SaveData("output.json",prob_mat, output);
	

	return 0;
} 