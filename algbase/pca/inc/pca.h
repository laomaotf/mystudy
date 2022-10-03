#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <memory>
namespace pca
{

	struct MODEL
	{
		std::vector<float> mean;
		std::vector<std::vector<float>> eigvecs;
	};

	std::shared_ptr<MODEL> BuildModel(std::vector< std::vector<float> >& datas);

	std::vector<float> Project(std::shared_ptr<MODEL> model, std::vector<float>& data, int N);

};