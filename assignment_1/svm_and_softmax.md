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

