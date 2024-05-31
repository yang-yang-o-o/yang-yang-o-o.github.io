---
title: "Deep Learning"
layout: post
date: 2022-12-22 10:01
# image: /assets/images/markdown.jpg
# headerImage: false
tag:
- CV
category: blog
# author: jamesfoster
# description: Markdown summary with different options
---

## 目录

- ### [1、数据](#customname1)

    - #### [1.1、数据预处理](#customname1_1)

    - #### [1.2、数据集划分](#customname1_2)

- ### [2、网络结构](#customname2)

    - #### [2.1、CNN](#customname2_1)

    - #### [2.2、FC](#customname2_2)

    - #### [2.3、Transformer](#customname2_3)

- ### [3、参数设置](#customname3)

- ### [4、损失函数](#customname4)

- ### [5、前向传播](#customname5)

- ### [6、反向传播](#customname6)

- ### [7、梯度下降](#customname7)

- ### [8、训练中](#customname8)

- ### [9、可视化](#customname9)

- ### [10、评估](#customname10)

- ### [11、算法调整](#customname11)

---

### 1、数据 {#customname1}

- #### 1.1、数据预处理 {#customname1_1}

    - ##### 数据增强

        - 畸变

            <!-- ![Markdowm Image](/assets\Deep_Learning\image_11.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_11.png)
        
        - 图像遮挡

            - 随机擦除

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_13.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_13.png)

            - Cutout

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_14.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_14.png)

            - Hide and Seek

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_15.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_15.png)

            - Grid Mask

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_16.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_16.png)

            - MixUp

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_17.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_17.png)

            - CutMix

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_18.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_18.png)

            - mosaic

                <!-- ![Markdowm Image](/assets\Deep_Learning\image_19.png) -->
                ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_19.png)

            - style transfer GAN
        - **数据增强方式有哪些？**
            畸变和图像遮挡。畸变又分为光照畸变和几何畸变，光照畸变比如改变HSV 等，几何畸变比如缩放、仿射变换，裁剪。图像遮挡有随机擦除、cutout、mixup、cutmix、mosaic。
            - 随机擦除：可以在输入图像上擦除、也可以在特征图上擦除，Dropout（随机失活神经元）、DropConnect（随机失活连接）、DropBlock（在卷积输出的特征图中随机将一个区域置0）
            - Cutout：随机左上角的一个固定大小的正方形区域置0.
            - Mixup：直接将两张图像利用α通道混合，拥有两个标签，loss 可以分别计算每个标签的loss，然后安装α加权。
            - CutMix：从另一张图像上裁剪一个区域贴到当前图像上，在区域中保留原图的bbx。
            - Mosaic：从4 张图像上裁剪得到4 个区域，然后拼接4 个区域得到一张图像，在区域中保留原图的bbx。好处是一张图像同时混合了4 中上下文信息，在BN 时就能同时考虑4 倍的信息，就能得到近似4 倍batch-size 的BN 效果。

    - ##### 归一化

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_12.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_12.png)

        - **输入数据为何要归一化、如何归一化？**
            当输入数据各个维度上的范围相差很大时，对输入数据归一化为0 均值1 方差，这样能加速收敛

- #### 1.2、数据集划分 {#customname1_2}

    <!-- ![Markdowm Image](/assets\Deep_Learning\image_20.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_20.png)

    - **什么是n 份交叉验证，有什么好处，如何用于模型评估？**
        - 是一种用于模型选择的策略，将数据集划分为n 份，对于一个模型训练n 次，每次使用其中1份作为验证集、其余n-1 份作为训练集，取n 次训练的平均误差。多个模型时根据平均误差来选择模型。这种策略的好处是可以避免固定划分数据集时的局限性、特殊性。尤其在小规模数据集上能减小过拟合。
        - 将这种策略用于训练集和测试集时，可以进行模型评估，每次用其中1 份作为测试集，n-1 份作为训练集，同一个模型训练n 次，取平均误差作为模型的评估结果。

### 2、网络结构 {#customname2}

- #### 2.1、CNN {#customname2_1}

    - 卷积

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_21.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_21.png)

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_22.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_22.png)

        - **传统卷积、Depthwise Convolution 、Pointwise Convolution 、Depthwise Separable Convolution、分组卷积、空洞卷积的具体做法？**
            - Depthwise Convolution：对每个通道单独卷积，卷积核个数为输入通道数，卷积前后通道数不变。不能融合不同通道的信息。
            - Pointwise Convolution：卷积核固定为1x1xC，就是1x1 卷积。
            - Depthwise Separable Convolution：先进行Depthwise Convolution，然后Pointwise Convolution。
            - 空洞卷积：在卷积核中插入空白数据，目的是增大感受野。
            - 分组卷积：将输入在通道上分为多个组，输出通道也分为多个组，每个组直接单独卷积。每个卷积核的尺寸变为（N/G，K，K），参数量变为原来的1/G。
        - **在计算机视觉领域，为什么卷积核的尺寸通常为奇数：**
            原因有三点：
            - 1、如果是一个偶数，那么same padding 时就只能非对称填充，奇数时直接填充卷积核的单边，更自然对称。
            - 2、奇数的卷积核有一个central pixel 可以方便的确定position。
            - 3、奇数相对于偶数，有中心，对边缘、对线条更加敏感，可以更有效的提取信息。如果使用偶数尺寸的卷积核应该也能得到不错的结果，但是在计算机视觉领域，通常使用奇数尺寸的卷积核。
        - **卷积的特点？**
            参数共享和稀疏连接。
            - 参数共享：卷积核在卷积的过程中，是整张图像共用的，卷积核的参数是在不同的卷积区域之间共享的。
            - 稀疏连接：卷积输出的某个值，只有对应的卷积区域有关，与其他区域无关，当前的值只连接到了这个区域，没有连接到其他区域。
        - **上采样分为哪几种？**
            UNpooling、转置卷积/反卷积；
            - UNpooling 又分为补0、补邻近值、max UNpooling（补0）
            - 转置卷积/反卷积：卷积核在输出图上滑动，将输入的一个值乘以卷积核得到输出图上的多个值，然后滑动过程中重叠位置相加。

    - BN

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_23.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_23.png)

        - **BN 的作用、训练时和推理时BN 的计算细节、BN 参数的维度。**
            BN 将卷积输出的尺度调整到统一的区间，减少数据的发散程度，降低网络的学习难度，加速收敛，减弱了前层参数和后层参数的联系，有轻微的正则效果，避免梯度消失和梯度爆炸。**BN 的精髓在于归一化后使用β和γ作为还原参数，用于保留原数据的分布。**
            - 训练阶段：计算当前batch 上的均值和方差，然后归一化，然后乘γ加β。然后用均值和方差去更新滑动平均值。
            - 推理阶段：使用训练阶段最后的滑动平均值作为均值和方差来归一化，然后用训练得到的γ和β来还原分布。
            <!-- ![Markdowm Image](/assets\Deep_Learning\image_27.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_27.png)
            - 参数维度：均值、方差、γ、β的维度都是卷积输出的通道数目，也就是说是在通道维度上BN。

    - 激活

        - 线性激活

        - 非线性激活
            Sigmoid、tanh、Relu、leak relu、maxout
            <!-- ![Markdowm Image](/assets\Deep_Learning\image_24.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_24.png)
        
        - dropout

            <!-- ![Markdowm Image](/assets\Deep_Learning\image_25.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_25.png)

            - **Dropout 的作用、训练和推理时的计算细节**
                能够消除特征间的相互适应，使模型在不需要所有特征都有时才能决策。训练时以一定的概率置0 一些元素，然后将未被置0 的每个元素都除以概率，保证当前层的期望不变。推理时不使用Dropout。

    - 池化

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_26.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_26.png)

- #### 2.2、FC {#customname2_2}

- #### 2.3、Transformer {#customname2_3}

### 3、参数设置 {#customname3}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_10.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_10.png)

### 4、损失函数 {#customname4}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_9.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_9.png)

- **L1 正则和L2 正则具体做法、为什么L1 正则比L2 正则稀疏？为什么正则项能减小过拟合？**
    L1 正则项为网络每个权重求L1 范数（矩阵中每个元素的绝对值之和），然后所有权重求和，损失函数中加入L1 正则项，损失函数对于某个权重的偏导数中就只有λsgn(w)这一项，乘以学习率得到ηλsgn(w)，根据权重的L1 范数的符号决定加减ηλ。无论当前权重的大小，总是加减一个固定的项。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_28.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_28.png)
    L2 正则项为网络每个权重求L2 范数（每个元素的平方和的平方根），然后所有权重求和，损失函数中加入L2 正则项，损失函数对于某个权重的偏导数中就只有ηλw 这一项，合并到w 中又叫权重衰减
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_29.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_29.png)
    权重每次乘以一个小于1 的数。因为L1 正则对于权重的影响每次都下降一个固定的值，因此不会保留很多很小的值，而是有很多为0 的值，因此得到的权重w 就稀疏。L2 正则对于权重的影响使得权重每次都乘以一个小于1 的数，当w 本身很小时，下降就变慢，最后不会完全等于0，因此有很多接近于0 的值，没有L1 稀疏。**正则项的本质就是通过减小权重w 来达到简化模型的目的，对于输入数据增加噪声之后，模型输出受影响的减小。**

- **Softmax+交叉熵损失求导过程？**
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_30.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_30.png)

### 5、前向传播 {#customname5}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_8.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_8.png)

- **滑动平均的优点、如何计算、偏差修正如何做？**
    使用滚动变量来求平均值，内存占用小，计算效率高。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_31.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_31.png)
    近似于前1/（1-β）个数据的平均值。偏差修正：
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_32.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_32.png)

### 6、反向传播 {#customname6}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_7.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_7.png)

- **标量、向量、矩阵间相互求导规则？**
    除了矩阵和行向量之间的导数是前者对于后者的每个元素分别求导，其他的都是前者的每个元素分别对于后者求导。
- **为什么会出现梯度消失和梯度爆炸？**
    因为当前层对于上一层的导数需要乘以当前层的权重，如果w 被调整到比单位矩阵小，那么对于反向传播的梯度是有缩小的作用，在深层网络中，如果梯度不断被缩小，就会出现梯度消失问题，如果w 被调整到比单位矩阵大，对于梯度有放大作用，就会出现梯度爆炸的问题。激活函数有时也会影响梯度，比如sigmoid 函数饱和时，梯度很小，最终如果w 也缩小梯度，那么就会出现梯度消失。
- **如何解决梯度消失和梯度爆炸？**
    正确选择激活函数、使用残差结构、使用BN

### 7、梯度下降 {#customname7}

- #### 定义

<!-- ![Markdowm Image](/assets\Deep_Learning\image_5.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_5.png)

- **为什么梯度的反方向是loss 下降最快的方向？**
    因为将损失函数在当前位置一阶泰勒展开，保留一阶项，一阶项是增量向量和梯度向量的点乘，等于模长的乘积乘以夹角的余弦值，当增量向量和梯度向量反向时，点乘最小，损失函数最小，此时增量向量的方向就是负梯度方向。

- #### 常用方法

<!-- ![Markdowm Image](/assets\Deep_Learning\image_6.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_6.png)

- **SGD、GD、mini-batch SGD 的区别？**
    mini-batch 既能加快收敛速度，所需要的硬件内存又不会过大。如果不考虑硬件，肯定是GD收敛速度最快。

- **SGDM、Adagrad、RMSProp、ADAM 各自具体的做法和优缺点？**
    SGDM 在SGD 的基础上用梯度的滑动平均值来代替梯度，引入之前梯度的影响，能加快收敛速度。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_33.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_33.png)
    Adagrad 累加梯度的平方，然后学习率除以其根植，以此来自适应调整学习率。当之前的梯度大时，分母大，学习率变小；当之前的梯度小时，分母小，学习率变大。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_34.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_34.png)
    RMSProp 在Adagrad 的基础上将梯度平方的累加和改为梯度平方的滑动平均值。也实现自适应调整学习率，但是很远的梯度就考虑得很少。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_35.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_35.png)
    ADAM 使用梯度平方的滑动平均来自适应调整学习率，同时使用梯度的滑动平均值作为梯度。并且使用偏差修正。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_36.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_36.png)

- **优化函数为什么不用sgd 而用adam，什么时候应该使用sgd，什么时候使用adam？**
    **ADAM 的收敛速度比SGDM 快，但是收敛效果没有SGDM 好**，因为ADMD 中使用一定窗口内梯度平方的滑动平均来自适应的调整学习率，如果数据发生巨变，那么学习率就时大时小，引起学习率震荡，导致模型最终收敛效果变差，如果最终的全局最优点是一个平坦的极值点，那么ADAM 会在收敛后尝试跳出，使得最终的收敛效果变差。前期梯度平方滑动平均小，学习率大，ADAM 收敛更快，ADAM 更倾向于收敛到sharp minimum，因为这种极值点附近的梯度大，进来时梯度平方的滑动平均大，学习率小，然后收敛于此处。如果处于平坦的极值点，梯度的平方小，学习率大，算法尝试跳出平坦的极值点。平坦的极值点通常训练集和测试集分布更接近，SGDM 的泛化能力强一些，ADAM 最终收敛的泛化能力差一些。

### 8、训练中 {#customname8}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_4.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_4.png)

- **Warm-up 作用？**
    在训练初期EMA 不稳定，此时让学习率小一些，待EMA 稳定以后，再选择预先设置的学习率进行训练
- **Early-stopping 作用？**
    有降低过拟合的作用，但是提早停止，代价函数的优化就停止了，偏差就不再下降。
- **Learning rate decay 作用**
    在初期设置较大的学习率，然后逐渐减小学习率能加快训练速度。可以根据公式衰减，可以根据epoch 衰减，可以按照指定的离散值衰减。
- **Shuffling 作用**
    避免模型从数据顺序中学习到偏见

### 9、可视化 {#customname9}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_3.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_3.png)

### 10、评估 {#customname10}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_2.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_2.png)

- **准确率、召回率、F1 分数？**
    True positive ：positive 表示当前预测为正样本，True 表示预测对了，标签也是正样本。
    <!-- ![Markdowm Image](/assets\Deep_Learning\image_37.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_37.png)
    F1 分数介于0-1 直接，越高越好。
- **偏差、方差和欠拟合、过拟合的关系？**

### 11、算法调整 {#customname11}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_1.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_1.png)

- **高偏差怎么调？**
    数据原因（特征不够）、网络结构原因（网络结构过于简单，使用更复杂的模型或者Boosting）、参数设置原因（学习率过小或过大）、损失函数原因（本身不收敛）、训练过程原因（训练次数太少，还没有收敛）  
    Boosting：将训练集分为多个子集，每个子集都训练一个简单的模型，最后加权所有模型。无论怎么的输入，总有一个模型能预测很好，最终加权输出也很好。

- **高方差怎么调？**
    数据原因（训练集和测试集分布差异大，用了过多不相关的特征。）、网络结构原因（网络结构过于复杂。简化结构、正则化、Dropout、BN、bagging）、训练过程原因（训练时间过长。early stopping）通常训练集和测试集分布差异大的原因是训练集太小。  
    Bagging：训练多个模型，然后预测时平均多个模型预测的结果。
