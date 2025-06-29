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