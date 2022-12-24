# 基于triplet loss的图像特征提取,支持图片搜索

* 使用hard sample容易的导致model collapse(现象是loss=margin),解决方法有
  1. 增加softmax loss做辅助
  2. 使用semi-hard sample替代hard sample做训练
