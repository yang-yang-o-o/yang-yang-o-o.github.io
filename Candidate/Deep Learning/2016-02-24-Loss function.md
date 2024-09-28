---
title: "Loss function"
layout: post
# date: 2016-02-24 22:48
# image: /assets/images/markdown.jpg
# headerImage: false
tag:
- Deep Learning
category: blog
# author: jamesfoster
# description: Markdown summary with different options
---

### MSE loss

$$
\mathrm{MSE}=\frac{1}{n} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2
$$

### CE loss

$$
\mathrm{CE}=-\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^C y_{i j} \log \left(\hat{y}_{i j}\right)
$$

### BCE loss

$$
\begin{aligned}
\operatorname{BCE}\left(p_{t}\right) & =-\log \left(p_{t}\right)
\end{aligned}
$$

$$
p_t=\left\{\begin{array}{lr}
p & \text { if } y=1 \\
1-p & \text { otherwise }
\end{array}\right.
$$

### BCE with logits

$$
\begin{aligned}
\operatorname{BCE}\left(p_{t}\right) & =-\log \left(p_{t}\right)
\end{aligned}
$$

$$
p_t=\left\{\begin{array}{lr}
sigmoid(p) & \text { if } y=1 \\
1-sigmoid(p) & \text { otherwise }
\end{array}\right.
$$

### Focal loss

$$
\begin{aligned}
\operatorname{FL}\left(p_{t}\right) & =-\alpha_{t}\left(1-p_{t}\right)^\gamma \log \left(p_{t}\right)
\end{aligned}
$$

$$
p_t=\left\{\begin{array}{lr}
p & \text { if } y=1 \\
1-p & \text { otherwise }
\end{array}\right.
$$

$$
\alpha_{t}=\left\{\begin{array}{lr}
\alpha & \text { if } y=1 \\
1-\alpha & \text { otherwise }
\end{array}\right.
$$

还可以看：

- 自己写的MR-Net.xmind
- OneNote
- Focal loss的论文

### IOU loss
