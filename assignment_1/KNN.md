# KNN 

## 基础知识

###### oad_CIFAR_batch

```python
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
```

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

###### cross validation

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)


# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}



################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
  k_to_accuracies[k]=[]#为每组k创建好列表
  #from collections import defaultdict
  #k_to_accuracies = defaultdict(list)避免显示判断key存在以及显示创建
  for i in range(num_folds):
    X_val=X_train_folds[i]#注意需要划分验证集

    y_val=y_train_folds[i]
    X_train_fold=np.vstack(X_train_folds[:i]+X_train_folds[i+1:])#上下堆叠
    y_train_fold=np.hstack(y_train_folds[:i]+y_train_folds[i+1:])#横向拼接
  
    classifier=KNearestNeighbor()
    classifier.train(X_train_fold,y_train_fold)
    dists=classifier.compute_distances_no_loops(X_val)
    y_pred=classifier.predict_labels(dists,k=k)#是利用dists矩阵预测标签
    accuracy=np.mean(y_pred==y_val)
    k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
```

[numpy.array_split — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)：划分数据集

*defaultdict避免显示创建key*

[numpy.vstack — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html)	[numpy.hstack — NumPy v2.3 Manual](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)：hs横向堆叠，vs上下叠加

sorted确保输出的**有序**

散点误差图

```python
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
```

[matplotlib.pyplot.scatter — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)：散点图绘制，其中[k] * len(accuracies)创建一个[k,……]列表与之对应

.item()返回一个（k，v）的迭代器，k为key，v为列表

[matplotlib.pyplot.errorbar — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html)：xerr，yerr分别是x，y对应的标准差