#pragma once
#include "basic_tree.h"

std::vector<float> CalcLoss(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, std::string loss_type);

std::map<int, std::vector<int>> SplitX(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices, int feat_index, float threshold);

std::map<int, float> CalcPred(std::vector<std::vector<basic_tree::VALUE>>& data, std::vector<int>& indices);

bool try_stop_split(std::vector<std::vector<basic_tree::VALUE>>& data, basic_tree::CNode& node, int max_depth, float min_std);
