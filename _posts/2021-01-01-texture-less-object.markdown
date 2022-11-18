---
title: "Texture-less object 6D pose estimation for Robotic Assembly"
layout: post
date: 2021-01-01 --:--
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

<p align="center">
<iframe width="760" height="515" src="https://www.youtube-nocookie.com/embed/b6RE1jMu0XI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

#### &emsp;&emsp;The video above demonstrates robotic assembly based on monocular camera, where the assembly clearance between shaft and hole is 1mm.

<p align="center">
<iframe width="760" height="515" src="https://www.youtube-nocookie.com/embed/S8Oy4uCkzzw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

#### &emsp;&emsp;The video above demonstrates pose tracking of mechanical parts, where the red dots show the visible edges of the object.

---

<!-- What has inside?

- Gulp
- BrowserSync
- Stylus
- SVG
- No JS
- [98/100](https://developers.google.com/speed/pagespeed/insights/?url=http%3A%2F%2Fsergiokopplin.github.io%2Findigo%2F) -->

### 2. Image Demo

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/Mechanical%20Parts.png?raw=true)

<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/112.png?raw=true" width = "60%" height = "60%"/>
</center>
<figcaption class="caption">Fig. 1</figcaption>

#### &emsp;&emsp;The Fig.1 shows the pose estimation results in open environment of our method。


<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/2.png?raw=true" width = "60%" height = "60%"/>
</center>

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/113.jpg?raw=true)
<figcaption class="caption">Fig. 2</figcaption>

#### &emsp;&emsp;The Fig.2 shows the pose estimation results under the different camera pose.


<center>
<img src="https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/4.png?raw=true" width = "60%" height = "60%"/>
</center>

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/4.jpg?raw=true)
<figcaption class="caption">Fig. 3</figcaption>

#### &emsp;&emsp;The Fig.3 shows the process of robot assembly of our method.

---

<!-- [Check it out](https://sergiokopplin.github.io/indigo/) here.
If you need some help, just [tell me](https://github.com/sergiokopplin/indigo/issues). -->
### 3. Key-inside

- Yolo
- Convolutional Autoencoder
- Edge distance tensor
- Nonlinear optimization
- Opencv
- Pinhole camera model
- OpenGL
- Rigid Object 6D pose description
- Lie groups and Lie algebras

---

### 4. Method

<!-- - #### Algorithm process -->

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/114.png?raw=true)
<figcaption class="caption">Fig. 4. Algorithm overview</figcaption>

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/124.png?raw=true)
<figcaption class="caption">Fig. 5. Visualization of the algorithm process</figcaption>

<!-- - #### Object detection

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/115.png?raw=true)
<figcaption class="caption">Fig. 6. The architecture of object detection network</figcaption>

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/118.png?raw=true)
<figcaption class="caption">Fig. 7. Synthetic training data for object detection networks</figcaption>

- #### Initial pose estimation

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/116.png?raw=true)
<figcaption class="caption">Fig. 8. Convolutional Auto-encoder Network</figcaption>

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/119.png?raw=true)
![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/120.png?raw=true)
![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/117.png?raw=true)

![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/121.png?raw=true)
![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/122.png?raw=true)
![Markdowm Image](https://github.com/yang-yang-o-o/yang-yang-o-o.github.io/blob/main/assets/images/123.png?raw=true) -->

---
### 5. Contrast


##### AAE (ECCV 2018): Sundermeyer M, Marton Z C, Durner M, et al. Implicit 3d orientation learning for 6d object detection from rgb images[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 699-715.
---

### 6. Paper

* #### [面向机器人精密操作的物体六自由度单目实时位姿估计方法研究 - 硕士学位论文](https://pan.baidu.com/s/1se1wJLHhyGKLw54SS35zAQ)
* #### code come soon.

---
