# KNN 

## 基础知识

###### 数据格式

```python
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
```

load_CIFAR_batch函数

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

此句分为三步，第一步将3072一维展为（3，32，32），后交换维度顺序，改为常见的HWC格式，最后将原来的uint8转换为float

###### 训练数据表格（idx）

```python
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)

samples_per_class = 7

for y, cls in enumerate(classes):

  idxs = np.flatnonzero(y_train == y)

  idxs = np.random.choice(idxs, samples_per_class, replace=False)

  for i, idx in enumerate(idxs):

​    plt_idx = i * num_classes + y + 1

​    plt.subplot(samples_per_class, num_classes, plt_idx)

​    plt.imshow(X_train[idx].astype('uint8'))

​    plt.axis('off')

​    if i == 0:

​      plt.title(cls)

plt.show()
```

enumerate遍历编号，**从0开始**

np.flatnonzero返回符合标签的所有索引，一维数组

np.random.choice(idxs, samples_per_class, replace=False)随机选取，不重复

plt_idx = i * num_classes + y + 1绘图的idx**从1开始**

###### compute distance

one loop

```python
def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        # 欧几里得距离 = sqrt(sum((x - y)^2))
        diff = self.X_train - X[i]           # broadcast subtraction
        dists[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
    return dists

```

axis=1**横向相加**，对列元素进行操作；axis=0**纵向相加**，对行元素进行操作

no loop
$$
\|x-y\|^2=\|x\|^2+\|y\|^2-2x\cdot y
$$

```python
def compute_distances_no_loops(self, X):
    # Step 1: 平方项
    test_sq = np.sum(X ** 2, axis=1, keepdims=True)            # shape: (num_test, 1)
    train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # shape: (1, num_train)

    # Step 2: 点积项
    dot_product = X @ self.X_train.T                       # shape: (num_test, num_train)

    # Step 3: 距离公式
    dists = np.sqrt(test_sq + train_sq - 2 * dot_product)

    return dists

```

通过转换平方差和的形式利用矩阵运算代替，keepdims=True表示形状维度不变（*T大写*）

###### predict labels

```python
def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        # Find the k nearest neighbors' indices
        nearest_idxs = np.argsort(dists[i])[:k]  # 返回前k个最小距离的索引
        closest_y = self.y_train[nearest_idxs]   # 获取这些索引对应的标签

        # Count label frequency and resolve ties by choosing the smallest label
        labels, counts = np.unique(closest_y, return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]    # 频率最高的标签，自动解决tie

        # 如果有平局，np.argmax 会返回第一个最大值出现的索引，即最小标签

    return y_pred

```

[numpy.argsort — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)：返回从小到大的**索引数组**，[:k]前k个索引

[numpy.argsort — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)：**返回排序后的唯一值**，return_counts=True返回唯一值出现的次数

[numpy.argmax — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)：返回最大值的**索引**

###### 比较矩阵

```python
difference = np.linalg.norm(dists - dists_one, ord='fro')

```

[numpy.linalg.norm — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

Frobenius 范数：
$$
\|A\|_F=\sqrt{\sum_{i,j}|A_{i,j}|^2}
$$

###### 计算函数运行时间

```python
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
```

