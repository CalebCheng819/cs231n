# notes

![img](https://pic1.zhimg.com/7c204cd1010c0af1e7b50000bfff1d8e_1440w.jpg)

###### 数据预处理

**零均值的中心化**：原始像素值（从0到255），根据训练集中所有的图像计算出一个平均图像值，然后每个图像都减去这个平均值，这样图像的像素值就大约分布在[-127, 127]之间了。下一个常见步骤是，让所有数值分布的区间变为[-1, 1]

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

# optimization

##### 梯度计算

###### 有限差值（近似值）

```python
def eval_numerical_gradient(f, x):
  """  
  一个f在x处的数值梯度法的简单实现
  - f是只有一个参数的函数
  - x是计算梯度的点
  """ 

  fx = f(x) # 在原点计算函数值
  grad = np.zeros(x.shape)
  h = 0.00001

  # 对x中所有的索引进行迭代
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # 计算x+h处的函数值
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # 增加h
    fxh = f(x) # 计算f(x + h)
    x[ix] = old_value # 存到前一个值中 (非常重要)

    # 计算偏导数
    grad[ix] = (fxh - fx) / h # 坡度
    it.iternext() # 到下个维度

  return grad
```

[numpy.nditer — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html)：多维数组迭代器 multi_index允许访问多维索引，readwrite允许读写

[numpy.nditer.iternext — NumPy v2.4.dev0 Manual](https://numpy.org/devdocs/reference/generated/numpy.nditer.iternext.html)：下一次迭代

实际中用**中心差值公式（**centered difference formula）效果较好


$$
[f(x+h)-f(x-h)] / 2 h
$$

###### 微分分析


$$
L_i=\sum_{j \neq y_i}\left[\max \left(0, w_j^T x_i-w_{y_i}^T x_i+\Delta\right)\right]
$$

$$
\nabla_{w_j} L_i=1\left(w_j^T x_i-w_{y_i}^T x_i+\Delta>0\right) x_i
$$



计算对w的导数，乘上loss，即为梯度

###### 小梯度批量下降

```python
# 普通的小批量数据梯度下降

while True:
  data_batch = sample_training_data(data, 256) # 256个数据
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # 参数更新
```

这个小批量数据就用来实现一个参数更新

##### 算法

###### 随机梯度下降

```python
# 普通更新
x += - learning_rate * dx
```

梯度是上升方向，而我们是最小化损失函数，所以是**负数**

###### 动量更新

```python
# 动量更新
v = mu * v - learning_rate * dx # 与速度融合
x += v # 与位置融合
```

初始化为0的v和超参数mu（动量，一般设置为0.9），在交叉验证中，mu会被设置成[0.5,0.9,0.95,0.99]中的一个

一个典型的设置是刚开始将动量设为0.5而在后面的多个周期（epoch）中慢慢提升到0.99。

###### Nesterov动量

![img](https://pica.zhimg.com/412afb713ddcff0ba9165ab026563304_1440w.png)

```python
x_ahead = x + mu * v
# 计算dx_ahead(在x_ahead处的梯度，而不是在x处的梯度)
v = mu * v - learning_rate * dx_ahead
x += v
```

核心思路是向前看，对dx求导替换为dx_ahead

实际实现时，为避免更新x

```python
v_prev = v # 存储备份
v = mu * v - learning_rate * dx # 速度更新保持不变
x += -mu * v_prev + (1 + mu) * v # 位置更新变了形式
```

###### 学习率

**随步数衰减**：每进行几个周期就根据一些因素降低学习率。典型的值是每过5个周期就将学习率减少一半，或者每20个周期减少到之前的0.1。在实践中可能看见这么一种经验做法：使用一个固定的学习率来进行训练的同时观察验证集错误率，每当验证集错误率停止下降，就乘以一个常数（比如0.5）来降低学习率。

**指数衰减**：
$$
\alpha=\alpha_0 e^{-k t}
$$
其中alfalfa，k是超参数，t是迭代次数（也可以使用周期作为单位）

**1/t衰减**：
$$
\alpha=\alpha_0 /(1+k t)
$$

##### 逐参数适应学习率方法

###### Adagrad

```python
# 假设有梯度和参数向量x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

cache的尺寸与梯度尺寸一样，**eps**（一般设为1e-4到1e-8之间）

###### RMSprop

```python
cache =  decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

dacay_rate常见范围[0.9,0.99,0.999]，防止cache变化激进

###### Adam

```python
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

论文中推荐的参数值**eps=1e-8, beta1=0.9, beta2=0.999**

完整的Adam更新算法也包含了一个偏置*（bias）矫正*机制，因为**m,v**两个矩阵初始为0，在没有完全热身之前存在偏差，需要采取一些补偿措施

###### extra

- 使用随机搜索（不要用网格搜索）来搜索最优的超参数。分阶段从粗（比较宽的超参数范围训练1-5个周期）到细（窄范围训练很多个周期）地来搜索。
- 进行模型集成来获得额外的性能提高。

# codes_softmax

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

最先划分**验证集**，不能让模型“看到”验证集标签用于训练，否则会泄露。

然后从**训练集中**随机抽取开发集和测试集

###### bias trick

```python
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
```

常用偏置技巧：在输入数据的最后一列加上常数1


$$
X^{\prime}=\operatorname{np} . \operatorname{hstack}([\mathrm{X}, \operatorname{np} . \operatorname{ones}((\mathrm{N}, 1))]) \Rightarrow \quad \text { shape: }(N, D+1)
$$

##### 计算dw

###### naive

loss function:


$$
L_i=-s_{y_i}+\log \left(\sum_{j=1}^C e^{s_j}\right)
$$


对权重W求导，分为两类：

1. j=yi（真实类别）
   $$
   \frac{\partial L_i}{\partial w_j}=-x_i+\frac{e^{s_j}}{\sum_k e^{s_k}} x_i=\left(p_j-1\right) x_i
   $$
   

2. j≠yi

   
   $$
   \frac{\partial L_i}{\partial w_j}=\frac{e^{s_j}}{\sum_k e^{s_k}} x_i=p_j x_i
   $$
   

   故：

   
   $$
   \frac{\partial L_i}{\partial w_j}= \begin{cases}\left(p_j-1\right) x_i & \text { if } j=y_i \\ p_j x_i & \text { if } j \neq y_i\end{cases}
   $$
   

```python
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    #print(W.shape)
    #print(X.shape)
    num_train = X.shape[0]
    for i in range(num_train):#在循环里面计算loss和dw，出循环后/num_train
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        for j in range(num_classes):
            if j==y[i]:
                #print(X[i].shape)
                dW[:,j]+=(p[j]-1)*X[i]
            else:
                dW[:,j]+=p[j]*X[i]
    
    dW=dW/num_train+2*reg*np.sum(W*W)


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    
        

    return loss, dW
```

X[i]的维度（D，），NumPy 的默认行为是：从二维数组中取一行时，会**降维**成**一维数组**。

###### vector

$$

\frac{\partial L}{\partial W}=X^T(P-Y)


其中：
P \in \mathbb{R}^{N \times C} ：每个样本对每类的预测概率，
Y \in \mathbb{R}^{N \times C} ：每个样本的 one－hot 标签
$$

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    scores=X@W
    scores-=np.max(scores,axis=1,keepdims=True)
    p=np.exp(scores)
    p/=np.sum(scores,axis=1,keepdims=True)
    correct_p=p[np.arrange(num_train),y]#高级索引
    loss=-np.sum(np.log(correct_p))
    loss=loss/num_train+reg*np.sum(X*X)
    

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dp=p
    dp[np.arrange(num_train),y]-=1
    dW=X.T.dot(dp)/num_train#不要忘记除以num_train
    dW+=2*reg*np.sum(X*X)
    return loss, dW

```

[numpy.max — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.max.html)：返回一条轴上的最大值，可以控制维度不变

[numpy.arange — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)：与range类似，可以搭配使用高级索引和操作（**不要写错成arrange**）

###### 梯度检查（gradient check）

```python
from cs231n.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)
```

lambda是匿名函数的关键字，[0]表示返回的是softmax_loss_naive第一个返回值

```python
def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
```

**tuple**：将每个维度的随机值转换成位置索引

###### save model

```python
def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = {"W": self.W}
      np.save(fpath, params)
      print(fname, "saved.")
```

__file_ _是python内置变量，表示当前脚本的位置

**os.path.dirname**:获取当前文件的目录名

../saved/进入上一级目录，并进入saved/

[numpy.random.choice — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)：路径+文件，保存为npz格式

###### 散点热度图

```python
# Visualize the cross-validation results
import math
import pdb

# pdb.set_trace()

x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.tight_layout(pad=3)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()
```

![img](data:image/png;base64,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7du+fbvxoiMiIiIiovqvdPl7Q9rVEzolZvb29lqvR48eXS3BEBERERERNUQ6JWZffPFFdcdBREREREQNlUwq2QxpV0/o3ffXr18/ZGRklCnPyspCv379jBETERERERE1IJIkM3irL/RelTE2NhaFhYVlyvPz83HgwAGjBEVERERERA0Ie8x0T8xOnTql+e+//voLKSkpmtcqlQp79uxB06ZNjRsdERERERFRA6BzYhYQEKB58Ft5QxYtLS3x8ccfGzU4IiIiIiKq/ySZDJIBzzEzpE1tpXNilpSUBCEEvL29cfToUbi4uGj2WVhYoHHjxpDL5dUSJBERERER1WOSVLIZ0q6e0Dkx8/T0BACo1epqC4aIiIiIiBogmQQY0vvVEOeYldq5c2e55ZIkQalUomXLlmjRokWVAyMiIiIiogaCPWb6J2ajRo2CJEkQQmiVl5ZJkoSePXtix44dcHR0NFqgRERERERE9ZXe/YV79+5F586dsXfvXmRmZiIzMxN79+5F165d8fPPP+P333/HnTt38PLLL1dHvEREREREVM+ULv5hyFZf6N1jNmvWLHz66afo3r27pqx///5QKpV47rnncPbsWSxfvhxPP/20UQMlIiIiIqJ6SpKVbIa0qyf0TswSExNhZ2dXptzOzg5///03AKBVq1ZIT0+venRERERERFT/SQY+YLoezTHTO8UMDAzEnDlzcPv2bU3Z7du38corr6Bz584AgEuXLsHDw8N4URIRERERUb0lSTKDt/pC7x6z9evXY+TIkWjWrJkm+bp27Rq8vb3x448/AgBycnLwxhtvGDdSIiIiIiKiekrvxMzX1xd//fUXfv31V1y8eFFTNmDAAMj+mXw3atQoowZJRERERET1mMzAoYwN+TlmACCTyTBo0CAMGjTI2PEQEREREVFDw8U/DEvMoqOjER0djbS0NKjVaq19n3/+uUGBREZGYt68eZg1axaWL19ebp1169bhyy+/xJkzZwCUzHd755130KVLF4POSUREREREtQAfMK3/4h+LFi3CwIEDER0djfT0dNy7d09rM8SxY8ewdu1a+Pv7V1ovNjYW48aNQ0xMDOLi4uDh4YGBAwfixo0bBp2XiIiIiIhqAZnM8K2e0LvHbM2aNdiwYQMmTJhglABycnIwfvx4rFu3Dm+//Xaldbds2aL1+rPPPsP333+P6Oho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kJCAAQMGYMmSJXqdIyMjA/369UPHjh1x/Phx7NmzB6mpqXjiiSc0dXJzcxEeHo7jx48jOjoaMpkMo0ePhlqt1jrWq6++ilmzZuHcuXMICQkBAMTExCAxMRExMTHYuHEjNmzYgA0bNlQa06JFi/DEE0/g1KlTGDJkCMaPH4+7d+8CAJKSkvDYY49h1KhR+PPPP/H888/j9ddf1+la33zzTYwePRqnT5/G008/DbVajWbNmuHbb7/FX3/9hQULFuC1117DN998AwB4+eWX8cQTT2DQoEG4desWbt26he7du6OoqAghISGwtbXFgQMHcOjQIdjY2GDQoEEoLCzU9dYTERGVTxARkUn16dNHzJo1SwghxMWLFwUAcejQIc3+9PR0YWlpKb755hshhBBjxowRQ4cO1TrG+PHjhb29fYXnSEpKEgDEyZMnhRBCLF68WAwcOFCrzrVr1wQAceHChXKPcfv2bQFAnD59WuuYy5cv16o3ceJE4enpKYqLizVljz/+uBgzZozmtaenp/jwww81rwGIN954Q/M6JydHABC7d+8WQggxd+5c0b59e63zvP766wKAuHfvXoXXDUC8+OKLFe4vNW3aNPHoo49qXcPIkSO16mzatEn4+voKtVqtKSsoKBCWlpYiKirqgecgIiKqDHvMiIhqkXPnzsHMzAxdu3bVlDk7O8PX1xfnzp0DAFy4cAFdunTRavff1w/y559/IiYmBjY2NprNz88PADTDFS9duoRx48bB29sbdnZ28PLyAgAkJydrHSsoKKjM8du1awe5XK553aRJE6SlpVUak7+/v+a/ra2tYWdnp2lz4cIFdO7cWau+rtdcXnyrVq1CYGAgXFxcYGNjg08//bTMdf3Xn3/+icuXL8PW1lZzz5ycnJCfn681xJOIiMgQXJWRiKgBysnJwfDhw/Huu++W2dekSRMAwPDhw+Hp6Yl169bB3d0darUa7du3LzNsr7zVFc3NzbVeS5JUZgikMdro4r/xbd26FS+//DI++OADdOvWDba2tnj//fdx5MiRSo+Tk5ODwMBAbNmypcw+FxeXKsdJREQNGxMzIqJapE2bNiguLsaRI0fQvXt3AMCdO3dw4cIFtG3bFgDg6+uLY8eOabX77+sH6dSpE77//nt4eXnBzKzsR0HpOdetW4devXoBAA4ePGjIJRmFr68vdu3apVWm7zWX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duwYUlNTtZuPjw9mzJiBHTt26L1Pg4cy/vOf/8SqVasQHx9vaFMiIiIiIqIqhCRByOgxk9MmJiYGEyZMQNeuXdG9e3ckJCSgsLAQ0dHRAICoqCj4+vpqJxCJi4tD165dERgYiJKSEmzbtg1r1qzBsmXLAABNmjRBkyZNdI5hbW0Nb29vBAUF6R2XwYlZeXk5PvvsM/z0008ICwuDg4ODzvsLFiwwdJdERERERGTJJEnm5B+GJ2Zjx47F9evXMWfOHFy7dg2hoaFITEzUTgiSkZEBxd964goLCzF58mRcunQJdnZ2CA4Oxtq1azF27FjD470HSQghDGlwvynxd+/eXauA6lp+fj5cXFyQl5dXpw8IEhERERHVt8Z2r1sZ77XvPoGzQ9W1ku/bvrAI3sOfazTney8G95g19MSLiIiIiIgamVpOl28ODD6TZ555Brdv365SXlhYiGeeecYoQRERERERkeWofMZMzmYuDE7MVq9ejTt37lQpv3PnDv7zn/8YJSgiIiIiIrIglT1mcjYzofdQxvz8fAghIITA7du3oVKptO+p1Wps27YNnp6edRIkERERERGZsXpax6wh0zsxc3V1hSRJkCQJbdu2rfK+JEmYO3euUYMjIiIiIiKyBHonZrt374YQAv3798c333wDd3d37Xs2Njbw9/eHj49PnQRJRERERERmTMZi0dp2ZkLvxKxv374AgPT0dLRo0QKSGXUbEhERERGR6dTnAtMNlcEp5qlTp7B3717t6yVLliA0NBRPPvkkbt68adTgiIiIiIjIAnDyD8MTsxkzZiA/Px8AcPz4ccTExGDo0KFIT09HTEyM0QMkIiIiIiLzJiSF7M1cGLzAdHp6Otq3bw8A+OabbzB8+HC89957OHLkCIYOHWr0AImIiIiIiMydwSmmjY0NioqKAAA//fQTBg0aBABwd3fX9qQRERERERHprXK6fDmbmTC4x6x3796IiYlBr169cPDgQWzcuBEAcObMGTRv3tzoARIRERERkXkTkDcsURjez9RgGXwmixcvhpWVFb7++mssW7YMvr6+AIDt27dj8ODBRg+QiIiIiIjMHHvMDO8xa9GiBb7//vsq5QsXLjRKQEREREREZGEkSd4Mi2aUmJlP3x8REREREVEjZXCPGRERERERkTFxgWkmZkREREREZGpyF4u25HXMiIiIiIiIjElA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9x3c3d0BAIWFhXBycqp9dEREREREZPYqe8zkbObC4KGMs2fPxvPPP4/du3drnzE7dOgQtm3bhuXLlwMAdu7cib59+xo3UiIiIiIiIjNlcGI2ceJEtG/fHosXL8amTZsAAEFBQfj555/Rs2dPAMD06dONGyUREREREZktjZA5Xb4l95gBQK9evdCrVy9jx0JERERERBaIk3/ITMzUajW2bNmCU6dOAQA6dOiAESNGQKlUGjU4IiIiIiIyf0IjIGR0f8lp01AZnJidO3cOQ4cOxeXLlxEUFAQAiIuLg5+fH3744QcEBgYaPUgiIiIiIiJzZvCsjFOnTkVgYCAyMzNx5MgRHDlyBBkZGWjZsiWmTp1aFzESEREREZEZq3zGTM5mLgzuMfv555/x22+/aafGB4AmTZogPj6ez50REREREZHB+IyZjMTM1tYWt2/frlJeUFAAGxsbowRFRERERESWQ6MR0Mjo/pLTpqEyeCjjP/7xDzz33HM4cOAAhBAQQuC3337DpEmTMGLEiLqIkYiIiIiIzBgXmJaRmH300UcIDAxEjx49oFKpoFKp0KtXL7Ru3RqLFi2qixiJiIiIiIjMmsFDGV1dXfHtt9/i7NmzOH36NACgXbt2aN26tdGDIyIiIiIi88dnzGSuYwYAbdq0QZs2bYwZCxERERERWSCNENDIyLLktGmo9ErMYmJi9N7hggULZAdDRERERESWR2gqNjntzIVeidnvv/+u184kSapVMEREREREZHkEKiYVlNPOXOiVmO3evbuu4yAiIiIiIrJYsp8xIyIiIiIiMgahATQcykhERERERGQ6lesjy2lnLpiYERERERGRSWlExSannblgYkZERERERCYlNAJCRpYlp01DpTB1AERERERERJaOPWZERERERGRSQlRsctqZC/aYERERERGRSWk0QvYmx5IlSxAQEACVSoXw8HAcPHiwxrorV67EQw89BDc3N7i5uSEiIqJK/YKCArzwwgto3rw57Ozs0L59eyxfvtygmEyamC1btgwhISFwdnaGs7MzevToge3bt9+zTUJCAoKCgmBnZwc/Pz+8/PLLKC4urqeIiYiIiIjI2CpnZZSzGWrjxo2IiYnBm2++iSNHjqBTp06IjIxEdnZ2tfWTk5Mxbtw47N69G/v374efnx8GDRqEy5cva+vExMQgMTERa9euxalTp/DS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)

1. x_scatter，y_scatter为画的点创建列表

2. maker_size=100表示点的大小，c是点的颜色数组，cmap是对应的颜色映射表

   `plt.cm.viridis`：绿色到黄色，推荐用于科学可视化（色盲友好）

   `plt.cm.plasma`：紫→黄

   `plt.cm.inferno`：黑→橙红

   `plt.cm.RdBu`：红蓝双向渐变

   `plt.cm.Greys`：灰度图

3. [matplotlib.pyplot.subplot — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html)：图的行数，列数，本张图对应的索引

4. [matplotlib.pyplot.tight_layout — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html)：pad给出图之间的间隔

5. [matplotlib.pyplot.colorbar — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html)：默认在右边给出颜色映射表

   ###### visualize weights

   ```python
   # Visualize the learned weights for each class.
   # Depending on your choice of learning rate and regularization strength, these may
   # or may not be nice to look at.
   w = best_softmax.W[:-1,:] # strip out the bias
   w = w.reshape(32, 32, 3, 10)
   w_min, w_max = np.min(w), np.max(w)
   classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   for i in range(10):
       plt.subplot(2, 5, i + 1)
   
       # Rescale the weights to be between 0 and 255
       wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
       plt.imshow(wimg.astype('uint8'))
       plt.axis('off')
       plt.title(classes[i])
   ```

   [:-1,:]去掉最后一行

   [numpy.squeeze — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)：去掉多余，即为1的维度

   

