---
title: "Linear kalman filter"
layout: post
date: 2022-02-27 22:48
# image: /assets/images/markdown.jpg
# headerImage: false
tag:
- SLAM
category: blog
# author: jamesfoster
# description: Markdown summary with different options
---

## 1、前言

卡尔曼滤波（Kalman Filtering）是一种用于估计系统状态的递归滤波技术。它通过将系统的动态模型和观测数据结合起来，根据过去的观测结果和状态估计，逐步更新系统的状态估计，从而提供对系统当前状态的最优估计。

卡尔曼滤波广泛应用于估计和控制问题，特别是在信号处理、机器人技术、导航系统、航天和航空领域等。它的主要优点是在存在噪声和不确定性的情况下，能够提供最优估计，并且具有良好的递归性能。

卡尔曼滤波基于以下两个基本假设：

- 系统模型假设：系统的状态可以用线性动态模型描述，并且受到高斯噪声的影响。这个模型通常由状态转移方程描述，可以预测系统在下一个时间步的状态。

- 观测模型假设：系统的状态只能通过受到高斯噪声影响的观测值来观测到。这个模型通常由观测方程描述，将系统状态映射到观测空间。

本文记录卡尔曼滤波的一种推导过程

## 2、卡尔曼滤波

### 2.1、基本动态系统模型

对于线性高斯系统，运动方程和观测方程可以通过线性方程来描述：
$$
\left\{\begin{array}{l}
\boldsymbol{x}_k=\boldsymbol{A}_k \boldsymbol{x}_{k-1}+\boldsymbol{B}_k \boldsymbol{u}_k+\boldsymbol{w}_k \\
\boldsymbol{z}_k=\boldsymbol{H}_k \boldsymbol{x}_k+\boldsymbol{v}_k
\end{array} \quad k=1, \ldots, N .\right.
\tag{1}
$$
卡尔曼滤波模型假设 k 时刻的真实状态 $\boldsymbol{x}_k$ 是从（k − 1）时刻的状态 $\boldsymbol{x}_{k-1}$ 演化而来，即式（1）中上面一个公式 $\boldsymbol{x}_k=\boldsymbol{A}_k \boldsymbol{x}_{k-1}+\boldsymbol{B}_k \boldsymbol{u}_k+\boldsymbol{w}_k$ ,其中：

- $\boldsymbol{A}_k$ 是作用在 $\boldsymbol{x}_{k-1}$ 上的状态变换模型（/矩阵/向量）。
- $\boldsymbol{B}_k$ 是作用在控制器向量 $\boldsymbol{u}_k$ 上的输入－控制模型。
- $\boldsymbol{w}_k$ 是过程噪声，并假定其符合均值为零，协方差矩阵为 $\boldsymbol{Q}_k$ 的多元正态分布。$\boldsymbol{w}_k \sim N\left(0, \boldsymbol{Q}_k\right)$

时刻k，对真实状态 $\boldsymbol{x}_k$ 的一个测量 $\boldsymbol{z}_k$ 满足式（1）中下面一个公式 $\boldsymbol{z}_k=\boldsymbol{H}_k \boldsymbol{x}_k+\boldsymbol{v}_k$ ,其中：

- $\boldsymbol{H}_k$ 是观测模型，它把真实状态空间映射成观测空间
- $\boldsymbol{v}_k$ 是观测噪声，其均值为零，协方差矩阵为 $\boldsymbol{R}_k$ ,且服从正态分布。$\boldsymbol{v}_k \sim N\left(0, \boldsymbol{R}_k\right)$

初始状态以及每一时刻的噪声 $\left\{\boldsymbol{x}_0, \boldsymbol{w}_1, \ldots, \boldsymbol{w}_k, \boldsymbol{v}_1 \ldots \boldsymbol{v}_k\right\}$ 都认为是互相独立的。

实际上，很多真实世界的动态系统都并不确切的符合这个模型；但是由于卡尔曼滤波器被设计在有噪声的情况下工作，一个近似的符合已经可以使这个滤波器非常有用了。

### 2.2、卡尔曼滤波器

卡尔曼滤波是一种递归的估计，即只要获知上一时刻状态的估计值以及当前状态的观测值就可以计算出当前状态的估计值，因此不需要记录观测或者估计的历史信息。

卡尔曼滤波器的状态由以下两个变量表示：

- $\hat{\boldsymbol{x}}_{k \mid k}$，在时刻k的状态的估计；
- $\boldsymbol{P}_{k \mid k}$，后验估计误差协方差矩阵，度量估计值的精确程度。

卡尔曼滤波器的操作包括两个阶段：**预测**与**更新**。在预测阶段，滤波器使用上一状态的估计，做出对当前状态的估计。在更新阶段，滤波器利用对当前状态的观测值优化在预测阶段获得的预测值，以获得一个更精确的新估计值。

- ### 预测

$$
\begin{aligned}
& \hat{\boldsymbol{x}}_{k \mid k-1}=\boldsymbol{A}_k \hat{\boldsymbol{x}}_{k-1 \mid k-1}+\boldsymbol{B}_k \boldsymbol{u}_k \tag{2}
\end{aligned}
$$

$$
\begin{aligned}
& \boldsymbol{P}_{k \mid k-1}=\boldsymbol{A}_k \boldsymbol{P}_{k-1 \mid k-1} \boldsymbol{A}_k^T+\boldsymbol{Q}_k \tag{3}
\end{aligned}
$$

- ### 更新

    - 测量残差
    $$\tilde{\boldsymbol{y}}_k=\boldsymbol{z}_k-\boldsymbol{H}_k \hat{\boldsymbol{x}}_{k \mid k-1}$$

    - 测量残差协方差
    $$\boldsymbol{S}_k=\boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T+\boldsymbol{R}_k$$

    - 最优卡尔曼增益
    $$\boldsymbol{K}_k=\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{S}_k^{-1} \tag{4}$$

    然后用它们来更新滤波器变量 $\boldsymbol{x}$ 与 $\boldsymbol{P}$ :

    - 更新的状态估计
    $$\hat{\boldsymbol{x}}_{k \mid k}=\hat{\boldsymbol{x}}_{k \mid k-1}+\boldsymbol{K}_k \tilde{\boldsymbol{y}}_k \tag{5}$$

    - 更新的协方差估计
    $$\boldsymbol{P}_{k \mid k}=\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right) \boldsymbol{P}_{k \mid k-1} \tag{6}$$

    使用上述公式计算 $\boldsymbol{P}_{k \mid k}$ 仅在最优卡尔曼增益的时候有效。使用其他增益的话，公式要复杂一些，参见 **2.3、公式推导**

- ### 不变量(Invariant)

    如果模型准确，而且 $\hat{\boldsymbol{x}}_{0 \mid 0}$ 与 $\hat{\boldsymbol{P}}_{0 \mid 0}$ 的值准确的反映了最初状态的分布，那么以下不变量就保持不变：所有估计的误差均值为零

    - $\mathrm{E}\left[\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k}\right]=\mathrm{E}\left[\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right]=0$
    - $\mathrm{E}\left[\tilde{\boldsymbol{y}}_k\right]=0$

    且协方差矩阵准确的反映了估计的协方差：

    - $\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k}\right)$

    - $\boldsymbol{P}_{k \mid k-1}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)$

    - $\boldsymbol{S}_k=\operatorname{cov}\left(\tilde{\boldsymbol{y}}_k\right)$

    请注意，其中 $\mathrm{E}\left[a\right]$ 表示 $a$ 的期望值 $\operatorname{cov}({\boldsymbol{a}})=\operatorname{E}[{\boldsymbol{a}}\boldsymbol{a}^{T}] $

### 2.3、公式推导

根据k-1时刻的状态，通过式(2)，可得到k时刻对状态的估计：
$$\hat{\boldsymbol{x}}_{k \mid k-1}=\boldsymbol{A}_k \hat{\boldsymbol{x}}_{k-1 \mid k-1}+\boldsymbol{B}_k \boldsymbol{u}_k \tag{7}$$

根据观测方程可得到k时刻状态的观测：
$$\boldsymbol{z}_{k}=\boldsymbol{H}_k \hat{\boldsymbol{x}}_{k \mid k-1} \rightarrow{\hat{\boldsymbol{x}}_{k \mid k-1 \mid measure}}={\boldsymbol{H}_k}^{-} \boldsymbol{z}_{k} \tag{8}$$

由式（7）和式（8）可构建k时刻状态的最终估计值
$$\hat{\boldsymbol{x}}_{k \mid k}=\hat{\boldsymbol{x}}_{k \mid k-1} + G * ({\hat{\boldsymbol{x}}_{k \mid k-1 \mid measure}} - \hat{\boldsymbol{x}}_{k \mid k-1}) \tag{9}$$
本质上，卡尔曼滤波是两个高斯分布的融合，一个是由运动方程得到的k时刻的状态 $\hat{\boldsymbol{x}}_{k \mid k-1}$，一个是由观测方程得到的k时刻的状态 $\hat{\boldsymbol{x}}_{k \mid k-1 \mid measure}$，融合后的k时刻状态称为后验估计

令 $ G =\boldsymbol{K}_{k} \boldsymbol{H}_{k}$，式（8）可变为
$$\hat{\boldsymbol{x}}_{k \mid k}=\hat{\boldsymbol{x}}_{k \mid k-1} + \boldsymbol{K}_{k}  (\boldsymbol{z}_{k} - \boldsymbol{H}_{k}\hat{\boldsymbol{x}}_{k \mid k-1}) \tag{10}$$

k时刻的实际状态为 $\boldsymbol{x}_{k}$，因此状态估计误差为
$$\boldsymbol{e}_{k} = \boldsymbol{x}_{k} - \hat{\boldsymbol{x}}_{k \mid k}$$

状态估计误差越小，表示最终估计值，或者说后验估计值越接近真实值，卡尔曼估计的目标就是求一个 $\boldsymbol{K}_{k}$ 使得状态估计误差最小

由2.2节中不变量部分可得 $\mathrm{E}\left[\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k}\right]=0$，$\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k}\right)$

因此 $\boldsymbol{P}_{k \mid k}$ 越小，表示最终估计值距离实际值的方差越小，而目标转化为求一个 $\boldsymbol{K}_{k}$ 使得 $\boldsymbol{P}_{k \mid k}$ 最小

#### 先验协方差矩阵 $\boldsymbol{P}_{k \mid k-1}$ 推导

在不变量中，先验协方差矩阵为：

$\boldsymbol{P}_{k \mid k-1}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)$

代入式（1）和式（2）得：

$\boldsymbol{P}_{k \mid k-1}=\operatorname{cov}\left(\boldsymbol{A}_k \boldsymbol{x}_{k-1}+\boldsymbol{B}_k \boldsymbol{u}_k+\boldsymbol{w}_k - \boldsymbol{A}_k \hat{\boldsymbol{x}}_{k-1 \mid k-1}-\boldsymbol{B}_k \boldsymbol{u}_k\right)$

化简得：
$\boldsymbol{P}_{k \mid k-1}=\operatorname{cov}\left(\boldsymbol{A}_k(\boldsymbol{x}_{k-1} - \hat{\boldsymbol{x}}_{k-1 \mid k-1}) +  \boldsymbol{w}_k\right)$

利用[协方差矩阵](https://zh.wikipedia.org/zh-hans/%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5)的性质 
$\operatorname{var}\left(\boldsymbol{a}^{\top} \boldsymbol{X}\right)=\boldsymbol{a}^{\top} \operatorname{var}(\boldsymbol{X}) \boldsymbol{a}$ 
$\operatorname{var}(\boldsymbol{X}+\boldsymbol{Y})=\operatorname{var}(\boldsymbol{X})+\operatorname{cov}(\boldsymbol{X}, \boldsymbol{Y})+\operatorname{cov}(\boldsymbol{Y}, \boldsymbol{X})+\operatorname{var}(\boldsymbol{Y})$
以及 $\boldsymbol{x}$ 和 $\boldsymbol{w}$ 相互独立，协方差矩阵为0，
$\boldsymbol{Q}_k = \operatorname{var}\left(\boldsymbol{w}_k\right)$
可得：

$\boldsymbol{P}_{k \mid k-1}=\boldsymbol{A}_k\operatorname{var}\left(\boldsymbol{x}_{k-1} - \hat{\boldsymbol{x}}_{k-1 \mid k-1}\right)\boldsymbol{A}_k^T +  \boldsymbol{Q}_k$

由于 $\boldsymbol{P}_{k-1 \mid k-1}=\operatorname{var}\left(\boldsymbol{x}_{k-1} - \hat{\boldsymbol{x}}_{k-1 \mid k-1}\right)$，因此

$$\boldsymbol{P}_{k \mid k-1}=\boldsymbol{A}_k \boldsymbol{P}_{k-1 \mid k-1} \boldsymbol{A}_k^T +  \boldsymbol{Q}_k$$


#### 后验协方差矩阵 $\boldsymbol{P}_{k \mid k}$ 推导

按照上边的定义，从误差协方差 $\boldsymbol{P}_{k \mid k}$ 开始推导：

$\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k}\right)$

代入式（10）得到：

$\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\boldsymbol{x}_k- (\hat{\boldsymbol{x}}_{k \mid k-1} + \boldsymbol{K}_{k}  (\boldsymbol{z}_{k} - \boldsymbol{H}_{k}\hat{\boldsymbol{x}}_{k \mid k-1}))\right)$

代入式（8）得到：

$
\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\boldsymbol{x}_k-\left(\hat{\boldsymbol{x}}_{k \mid k-1}+\boldsymbol{K}_k\left(\boldsymbol{H}_k \boldsymbol{x}_k+\boldsymbol{v}_k-\boldsymbol{H}_k \hat{\boldsymbol{x}}_{k \mid k-1}\right)\right)\right)
$

整理得到：
$\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right)\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)-\boldsymbol{K}_k \boldsymbol{v}_k\right)$

因为测量误差 $\boldsymbol{v}_k$ 与其他项是非相关的，因此有：

$\boldsymbol{P}_{k \mid k}=\operatorname{cov}\left(\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right)\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)\right)+\operatorname{cov}\left(\boldsymbol{K}_k \boldsymbol{v}_k\right)$

利用[协方差矩阵](https://zh.wikipedia.org/zh-hans/%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5)的性质 $\operatorname{var}\left(\boldsymbol{a}^{\top} \boldsymbol{X}\right)=\boldsymbol{a}^{\top} \operatorname{var}(\boldsymbol{X}) \boldsymbol{a}$ 可得：

$\boldsymbol{P}_{k \mid k}=\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right) \operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right)^T+\boldsymbol{K}_k \operatorname{cov}\left(\boldsymbol{v}_k\right) \boldsymbol{K}_k^T$ 

使用不变量 $\boldsymbol{P}_{k \mid k-1}=\operatorname{cov}\left(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k \mid k-1}\right)$ 以及 $\boldsymbol{R}_k$ 的定义得：

$$\boldsymbol{P}_{k \mid k}=\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right) \boldsymbol{P}_{k \mid k-1}\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right)^T+\boldsymbol{K}_k \boldsymbol{R}_k \boldsymbol{K}_k^T \tag{11}$$

式（11）给出了 $\boldsymbol{K}_{k}$ 和 $\boldsymbol{P}_{k \mid k}$ 的关系，该式对于任何卡尔曼增益 $\boldsymbol{K}_{k}$ 都成立。

而最终的目标是求一个最优的卡尔曼增益 $\boldsymbol{K}_{k}$ 使得 $\boldsymbol{P}_{k \mid k}$ 最小

#### 最优卡尔曼增益 $\boldsymbol{K}_{k}$ 推导

由于协方差矩阵是一个对称矩阵，其中对角线上的元素表示每个随机变量的方差，非对角线上的元素表示随机变量之间的协方差。当协方差矩阵是对角矩阵时，非对角线上的协方差为零，意味着不同维度的随机变量之间没有线性相关性，它们是相互独立的。

协方差矩阵的迹是指对角线上元素的和，即所有随机变量的方差之和。迹的大小可以用来衡量协方差矩阵的大小。当协方差矩阵的迹较大时，表示随机变量在不同维度上的方差较大，即各个维度的随机变量波动较大。相反，当协方差矩阵的迹较小时，表示随机变量在不同维度上的方差较小，即各个维度的随机变量波动较小。

当 $\boldsymbol{P}_{k \mid k}$ 的迹最小时，得到最优的卡尔曼增益 $\boldsymbol{K}_{k}$

令 $\boldsymbol{S}_k=\boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T+\boldsymbol{R}_k$ ，式（11）可以写为

$$
\begin{aligned}
\boldsymbol{P}_{k \mid k} & =\boldsymbol{P}_{k \mid k-1}-\boldsymbol{K}_k \boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1}-\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{K}_k^T+\boldsymbol{K}_k\left(\boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T+\boldsymbol{R}_k\right) \boldsymbol{K}_k^T \\
& =\boldsymbol{P}_{k \mid k-1}-\boldsymbol{K}_k \boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1}-\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{K}_k^T+\boldsymbol{K}_k \boldsymbol{S}_k \boldsymbol{K}_k^T \tag{12}
\end{aligned}
$$

当矩阵导数是 0 的时候得到 $\boldsymbol{P}_{k \mid k}$ 的迹（trace）的最小值：

$$
\frac{d \operatorname{tr}\left(\boldsymbol{P}_{k \mid k}\right)}{d \boldsymbol{K}_k}=-2\left(\boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1}\right)^T+2 \boldsymbol{K}_k \boldsymbol{S}_k=0 \tag{13}
$$

此处用到矩阵导数公式

$\frac{d \operatorname{tr}(\boldsymbol{B A C})}{d \boldsymbol{A}}=B^T C^T$

$\frac{d \operatorname{tr}(\boldsymbol{A B A^{\top}})}{d A}=2 A B$

$\operatorname{tr}(A)=\operatorname{tr}\left(A^{\mathrm{T}}\right)$

$\operatorname{tr}(m \boldsymbol{A}+n \boldsymbol{B})=m t r(\boldsymbol{A})+n t r(\boldsymbol{B})$

以及协方差矩阵的转置等于它本身
$\boldsymbol{P}_{k \mid k}^{\top} = \boldsymbol{P}_{k \mid k}$

从式（13）可以解出最优卡尔曼增益 $\boldsymbol{K}_{k}$ 

$\boldsymbol{K}_k \boldsymbol{S}_k=\left(\boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1}\right)^T=\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T$

$$\boldsymbol{K}_k=\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{S}_k^{-1} \tag{14}$$

当 $\boldsymbol{K}_{k}$ 为最优卡尔曼增益时，式（11）给出的后验协方差矩阵 $\boldsymbol{P}_{k \mid k}$ 可以继续化简，式（14）两边乘以 $\boldsymbol{S}_k \boldsymbol{K}_k{ }^T$ 得到

$$\boldsymbol{K}_k \boldsymbol{S}_k \boldsymbol{K}_k^T=\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{K}_k^T \tag{15}$$

根据式（12）

$$\boldsymbol{P}_{k \mid k}=\boldsymbol{P}_{k \mid k-1}-\boldsymbol{K}_k \boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1}-\boldsymbol{P}_{k \mid k-1} \boldsymbol{H}_k^T \boldsymbol{K}_k^T+\boldsymbol{K}_k \boldsymbol{S}_k \boldsymbol{K}_k^T$$ 

代入式（15）得

$$
\begin{aligned}
\boldsymbol{P}_{k \mid k} & =\boldsymbol{P}_{k \mid k-1}-\boldsymbol{K}_k \boldsymbol{H}_k \boldsymbol{P}_{k \mid k-1} \\
& =\left(I-\boldsymbol{K}_k \boldsymbol{H}_k\right) \boldsymbol{P}_{k \mid k-1} \tag{16}
\end{aligned}
$$

这个公式的计算比较简单，所以实际中总是使用这个公式，但是需注意这公式仅在使用最优卡尔曼增益的时候它才成立。如果算术精度总是很低而导致数值稳定性出现问题，或者特意使用非最优卡尔曼增益，那么就不能使用这个简化；必须使用上面导出的后验误差协方差公式。

## 参考
https://zh.wikipedia.org/zh-hans/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2#%E6%8E%A8%E5%AF%BC
https://www.bilibili.com/video/BV1ez4y1X7eR/?spm_id_from=333.788
https://zh.wikipedia.org/zh-hans/%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5
http://rosen.xyz/2017/03/04/%E7%9F%A9%E9%98%B5%E6%B1%82%E5%AF%BC%E6%96%B9%E6%B3%95/
https://zhuanlan.zhihu.com/p/273729929
https://zhuanlan.zhihu.com/p/514343061
https://www.cnblogs.com/sbb-first-blog/p/16583418.html