# notes

## Support Vector Machine

###### loss function


$$
L_i =\sum_{j\neq y_i}max(0,s_j-s_{y_i}+\Delta)= \sum_{j \neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)
$$


hinge loss（折页损失）：考虑线性评分函数，非正确答案评分减去正确答案评分加上Δ（margin）

![img](https://cs231n.github.io/assets/margin.jpg)

平方折页损失（L2-hinge loss）：更强烈的惩罚过界的边界值


$$
max(0,-)^2
$$

###### regularization正则化

$$
L=\underbrace{\frac{1}{N}\sum_iL_i}_{\mathrm{data~loss}}+\underbrace{\lambda R(W)}_{\textit{regularization loss}}=\frac{1}{N}\sum_i\sum_{j\neq y_i}[max(0,f(x_i;W)_j-f(x_i;W)_{y_i}+\Delta)]+\lambda\sum_k\sum_lW_{k,l}^2
$$

regularization penalty：为权重（非唯一）添加一些偏好，偏向于更小更分散的向量

## softmax

###### loss function


$$
L_i=-\log{\left(\frac{e^{f_{y_i}}}{\sum_je^{f_j}}\right)}\text{ or equivalently }\quad L_i=-f_{y_i}+\log\sum_je^{f_j}
$$


因为loss函数中含有指数项，数值计算较大，通过分子分母同乘一个常数C。就是应该将f的值同时减去一个常数，使得最大值为0


$$
\frac{e^{f_{y_{i}}}}{\sum_{j}e^{f_{j}}}=\frac{Ce^{f_{y_{i}}}}{C\sum_{j}e^{f_{j}}}=\frac{e^{f_{y_{i}}+\log C}}{\sum_{j}e^{f_{j}+\log C}}
$$


正则化参数的影响：随着正则化参数λ不断增强，权重数值会越来越小，最后输出的概率会接近于**均匀分布**。这就是说，softmax分类器算出来的概率最好是看成一种对于**分类正确性的自信**。

# codes

###### 数据集的划分

| 变量名           | 中文名 | 用途                              | 使用时间      |
| ---------------- | ------ | --------------------------------- | ------------- |
| `num_training`   | 训练集 | 拿来训练模型                      | 最开始        |
| `num_validation` | 验证集 | 拿来调参/选择模型结构             | 训练过程中    |
| `num_test`       | 测试集 | 拿来评估最终模型表现              | 最后阶段      |
| `num_dev`        | 开发集 | 训练前/调试时用来快速测试训练流程 | 开发/调试阶段 |

```python
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500
# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
```

最先划分验证集，不能让模型“看到”验证集标签用于训练，否则会泄露。