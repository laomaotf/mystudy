# dodgeball

## 重要参数

1. 模型感受野不大于resize比例: env_obs_resize_ratio,backbone
2. 增大每次采样后训练iterations: steps_each_collect

## 实验结果
### steps_each_collect = 100
![](results/dodgeball/resize4_100steps.jpg)
![](results/dodgeball/resize8_100steps.jpg)
![](results/dodgeball/resize16_100steps.jpg)
![](results/dodgeball/resize32_100steps.jpg)

### steps_each_collect = 512
![](results/dodgeball/resize4_512steps.jpg)
![](results/dodgeball/resize8_512steps.jpg)
![](results/dodgeball/resize16_512steps.jpg)


