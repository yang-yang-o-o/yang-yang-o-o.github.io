---
title: "A Patch Based Real-Time 6D Object Pose Refinement Method for Robotic Manipulation"
layout: post
date: 2020-07-01 --:--
tag: 6D pose
image: https://sergiokopplin.github.io/indigo/assets/images/jekyll-logo-light-solid.png
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "This is a simple and minimalist template for Jekyll for those who likes to eat noodles."
category: project
author: yang yang
externalLink: false
---
<!-- ![Screenshot](https://raw.githubusercontent.com/sergiokopplin/indigo/gh-pages/assets/screen-shot.png)

Example of project - Indigo Minimalist Jekyll Template - [Demo](https://sergiokopplin.github.io/indigo/). This is a simple and minimalist template for Jekyll for those who likes to eat noodles. -->

### 1. Video Demo
<iframe src="https://player.bilibili.com/player.html?aid=689871603&bvid=BV1o24y1f7NQ&cid=883264732&page=1&high_quality=1&danmaku=0" allowfullscreen="allowfullscreen" width="100%" height="500" scrolling="no" frameborder="0" sandbox="allow-top-navigation allow-same-origin allow-forms allow-scripts" high_quality="1"></iframe>

#### &emsp;&emsp;&emsp;&emsp; Link: [A Patch Based Real-Time 6D Object Pose Refinement Method for Robotic Manipulation](https://www.bilibili.com/video/BV1o24y1f7NQ/?share_source=copy_web&vd_source=926e5fb00a879a3a9c35633c5af54c69)


#### &emsp;&emsp;The above video shows the test results of our pose refinement method in the test environment and the open environment, where the initial, refined and Ground-Truth 3D Bounding Box are shown in orange, blue and green, respectively.
---
### 2. Image Demo

<!-- <embed src="Figure_21.pdf"
 type="application/pdf"  width="800px" height="2400px">

<iframe src="Figure_21.pdf"
 width="800px" height="2400px"></iframe>

<object data="Figure_21.pdf"
 type="application/pdf" width="800px" height="2400px"></object> -->

![Markdowm Image](https://raw.githubusercontent.com/yang-yang-o-o/yang-yang-o-o.github.io/main/assets/images/Figure_21.jpg)
<figcaption class="caption">Fig. 1. Pose refinement visualization on test data.</figcaption>

#### &emsp;&emsp;The Fig.1 shows the Pose refinement visualization of our method, where the refined and Ground-Truth 3D Bounding Box are shown in blue and green, respectively.

<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/111.png?raw=true" width = "60%" height = "60%"/>
</center>

<!-- ![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/111.png?raw=true){: class="center-image" } -->
<figcaption class="caption">Fig. 2. Robotic manipulation platform of Eye-in-Hand.</figcaption>


![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/Figure_12.jpg?raw=true)
<figcaption class="caption">Fig. 3. The pose estimation results visualization of Eye-in-Hand.</figcaption>

#### &emsp;&emsp;As shown in Fig. 3, we fix the relative pose between the calibration board and the object to evaluate the pose refinement accuracy under various camera view-points.

---
<!-- What has inside?

- Gulp
- BrowserSync
- Stylus
- SVG
- No JS
- [98/100](https://developers.google.com/speed/pagespeed/insights/?url=http%3A%2F%2Fsergiokopplin.github.io%2Findigo%2F) -->
### 3. Key-inside

- Siamese neural network
- Image patch matching
- PnP algorithm
- Opencv
- Pinhole camera model
- Rigid Object 6D pose description

---

### 4. Method

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/127.jpg?raw=true)

---

### 5. Contrast

<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/128.png?raw=true" width = "100%" height = "100%"/>
</center>

<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/129.png?raw=true" width = "60%" height = "60%"/>
</center>

##### YOLO6D (CVPR 2018): B. Tekin, S. N. Sinha, and P. Fua, “Real-time seamless single shot 6d object pose prediction,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 292–301

##### CullNet (ICCVW 2019): K. Gupta, L. Petersson, and R. Hartley, “Cullnet: Calibrated and pose aware confidence scores for object pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2019, pp. 0–0

##### PVNet (CVPR 2019): S. Peng, Y. Liu, Q. Huang, X. Zhou, and H. Bao, “Pvnet: Pixel-wise voting network for 6dof pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4561–4570

---
<!-- [Check it out](https://sergiokopplin.github.io/indigo/) here.
If you need some help, just [tell me](https://github.com/sergiokopplin/indigo/issues). -->
### 6. Paper

* #### [A Patch Based Real-Time 6D Object Pose Refinement Method for Robotic Manipulation](https://pan.baidu.com/s/1se1wJLHhyGKLw54SS35zAQ), password: yang
* #### code come soon.

---