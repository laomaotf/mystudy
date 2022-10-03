
# 逻辑回归LOSS：Negative Log Cross Entropy Loss
以二分类为例，输入 $(x_i,y_i),i=1,2,3...,m$，其中$y_i \in \{0,1\}$

$$
NLCELoss = yln \bar y + (1-y)ln(1-\bar y)
$$

其中$\bar y = \frac{1}{1+e^{-w^Tx+b}}$

$w=(w_1,w_2,w_3...,w_m)$, b是标量

$$
L(w,b)=yln \frac{1}{1+e^{-w^Tx+b}} - (1-y)ln(1 - \frac{1}{1+e^{-w^Tx+b}})
$$

损失函数L(w,b)对(w,b)求导

$$
\frac{\partial L}{\partial w} = \frac{x(\bar y - y)}{m}
$$

$$
\frac{\partial L}{\partial b} = \frac{\sum_{i=1}^m(\bar y - y)}{m}
$$

在SGD算法中调用上述梯度即可求解最优(w,b)