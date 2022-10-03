
# 线性拟合梯度公式
输入$(x_i,y_i), i=1,2,3,...m$
$$
L(w,b) = min \sum_{i=1}^m(f(x_i) - y_i)^2
$$

采用线性拟合$f(x) = wx + b$, 其中  
$w = (w_1,w_2,w_3,...,w_m)$，b是常量

$$
L(w,b) = min \sum_{i=1}^m(wx_i + b - y_i)^2
$$

计算Loss对(w,b)的梯度

$$
\frac{\partial L_{w,b}}{\partial w_j} = 2(w_j\sum_{i=1}^m{x_i^2} - \sum_{i=1}^m(y_i-b)x_i)
$$


$$
\frac{\partial L_{w,b}}{\partial b} = 2(mb - \sum_{i=1}^m(y_i-wx_i))
$$

上述是关于w和b的迭代公式，可以在SGD中调用。

令梯度为零，直接求最优(w,b) 

$$
w = \frac{\sum_{i=1}^my_i(x_i-\bar x)}{\sum_{i=1}^mx_i^2 - m\bar x^2}
$$

$$
b = \frac{1}{m}\sum_{i=1}^m(y_i - wx_i)
$$
上述公式可以在LSM中调用