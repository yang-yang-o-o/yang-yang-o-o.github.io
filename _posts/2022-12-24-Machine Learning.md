---
title: "Machine Learning"
layout: post
date: 2022-12-24 23:38
# image: /assets/images/markdown.jpg
# headerImage: false
tag:
- CV
category: blog
# author: jamesfoster
# description: Markdown summary with different options
---

## 目录

- ### [1、K-means](#customname1)

- ### [2、PCA](#customname2)

- ### [3、SVD](#customname3)

- ### [4、策略](#customname3)

---

### 1、K-means {#customname1}

<!-- ![Markdowm Image](/assets\Machine_Learning\image_1.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Machine_Learning\image_1.png)

- **KNN**
    计算当前样本和数据库中所有样本的距离，然后将距离排序，取最小的k 个距离对应的数据库样本，分析k 个样本的类别，取类别频率最高的类别作为当前样本的类别。
- **K-means**
    创建k 个初始聚类中心，每个数据点都分别计算到k 个聚类中心的距离，然后将每个数据点分到距离最近的聚类中心上，得到k 堆，然后计算每堆的均值作为新的聚类中心。然后重复上面的步骤，直到每个样本的类别都不改变。

### 2、PCA {#customname2}

<!-- ![Markdowm Image](/assets\Machine_Learning\image_2.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Machine_Learning\image_2.png)

当我们用一些特征来表示数据时，特征间可能出现冗余，此时可以通过PCA 将数据的特征降维。以实现用更低维度、更少、更显著的特征来表示数据。通常的做法先将数据归一化为0 均值1 方差，然后求数据的平方和矩阵
<!-- ![Markdowm Image](/assets\Machine_Learning\image_4.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Machine_Learning\image_4.png)
，对这个矩阵奇异值分解，U 矩阵的列向量是源数据列空间的标准正交基，取前k 个列向量就构成一个子空间，然后数据左乘k 个列向量构成的矩阵，就将源数据投影到了子空间，实现数据的降维。

### 3、SVD {#customname3}

<!-- ![Markdowm Image](/assets\Machine_Learning\image_3.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Machine_Learning\image_3.png)

- 只有方阵才能进行特征分解，特征分解将矩阵分解为特征向量张成的矩阵和特征值构成的对角阵的乘积。  
- 奇异值分解不要求矩阵为方阵，一个mxn 的矩阵可以分解为mxm 的酉矩阵、mxn 的对角阵、nxn 的酉矩阵三者的乘积。对角阵中对角元为奇异值。共轭转置等于逆矩阵称为酉矩阵。酉矩阵中每个元素都为实数变为正交矩阵。

- https://zhuanlan.zhihu.com/p/29846048

### 4、策略 {#customname4}

- 一开始先从简单的算法快速暴力的开始：通过分析算法的误差决定下一步做什么，通过数字评估指标来验证你想加入算法中的新想法是否能提高效果，进一步决定算法应该包含什么而不应该包含什么。

- 一种简单的查看描述特征是否足够的方法就是想想这些特征给人类的专家，其是否能预测出正确的结果。

- 在数据增强时要考虑的最重要的一点就是增强后的数据有没有代表性，也就是对最终提高模型的性能有没有很大的作用。
