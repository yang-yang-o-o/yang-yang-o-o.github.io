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

    - ##### 归一化

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_12.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_12.png)

- #### 1.2、数据集划分 {#customname1_2}

    <!-- ![Markdowm Image](/assets\Deep_Learning\image_20.png) -->
    ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_20.png)

### 2、网络结构 {#customname2}

- #### 2.1、CNN {#customname2_1}

    - 卷积

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_21.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_21.png)

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_22.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_22.png)

    - BN

        <!-- ![Markdowm Image](/assets\Deep_Learning\image_23.png) -->
        ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_23.png)

    - 激活

        - 线性激活

        - 非线性激活

            <!-- ![Markdowm Image](/assets\Deep_Learning\image_24.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_24.png)
        
        - dropout

            <!-- ![Markdowm Image](/assets\Deep_Learning\image_25.png) -->
            ![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_25.png)

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

### 5、前向传播 {#customname5}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_8.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_8.png)

### 6、反向传播 {#customname6}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_7.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_7.png)

### 7、梯度下降 {#customname7}

- #### 定义

<!-- ![Markdowm Image](/assets\Deep_Learning\image_5.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_5.png)

- #### 常用方法

<!-- ![Markdowm Image](/assets\Deep_Learning\image_6.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_6.png)

### 8、训练中 {#customname8}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_4.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_4.png)

### 9、可视化 {#customname9}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_3.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_3.png)

### 10、评估 {#customname10}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_2.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_2.png)

### 11、算法调整 {#customname11}

<!-- ![Markdowm Image](/assets\Deep_Learning\image_1.png) -->
![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/Deep_Learning\image_1.png)
