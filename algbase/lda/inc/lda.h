#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <memory>
namespace lda
{


	struct MODEL
	{
		std::vector<std::vector<float>> means;
		std::vector<float> w;
	};


	std::shared_ptr<MODEL> BuildModel(std::vector<std::vector< std::vector<float> >>& datas);

	float Project(std::shared_ptr<MODEL> model, std::vector<float>& data);

};