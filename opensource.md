---
title: Projects
layout: page
---

# <center> Projects

---

<div style="line-height: 1.2; display: flex; flex-direction: column; gap: 16px;">
    <ul style="margin: 0; padding-left: 20px;">
        <li>
            <h4 style="margin: 0;">
                Mono3D: 3D Object Detection for Autonomous Driving.<br>
                [<a id="repo-link" href="https://github.com/yang-yang-o-o/Mono3D" target="_blank" style="color: blue; text-decoration: underline;">Opensource</a>]
                <span id="star-count" style="font-size: 0.9em; color: gray;"></span><br>
                <span style="display: block; text-align: right; color: red;">3D Object Detection</span>
            </h4>
            <!-- <p style="text-indent: 2em; font-style: italic; margin: 0;">
                <strong>I led the design of the project, developing the entire algorithm framework, verifying its feasibility.</strong>
            </p> -->
        </li>
    </ul>
</div>

<div style="line-height: 1.2; display: flex; flex-direction: column; gap: 16px;">
    <ul style="margin: 0; padding-left: 20px;">
        <li>
            <h4 style="margin: 0;">
                EPRT：edge optimization based 6D Object pose refinement and tracking.<br>
                [<a id="repo-link" href="https://github.com/yang-yang-o-o/EPRT" target="_blank" style="color: blue; text-decoration: underline;">Opensource</a>]
                <span id="star-count" style="font-size: 0.9em; color: gray;"></span><br>
                <span style="display: block; text-align: right; color: red;">6D Object Pose Refinement &amp; Tracking</span>
            </h4>
            <!-- <p style="text-indent: 2em; font-style: italic; margin: 0;">
                <strong>I led the design of the project, developing the entire algorithm framework, verifying its feasibility.</strong>
            </p> -->
        </li>
    </ul>
</div>

<div style="line-height: 1.2; display: flex; flex-direction: column; gap: 16px;">
    <ul style="margin: 0; padding-left: 20px;">
        <li>
            <h4 style="margin: 0;">
                YOLO6D：RGB based 6D Object pose estimation.<br>
                [<a id="repo-link" href="https://github.com/yang-yang-o-o/YOLO6D" target="_blank" style="color: blue; text-decoration: underline;">Opensource</a>]
                <span id="star-count" style="font-size: 0.9em; color: gray;"></span><br>
                <span style="display: block; text-align: right; color: red;">6D Object Pose Estimation</span>
            </h4>
            <!-- <p style="text-indent: 2em; font-style: italic; margin: 0;">
                <strong>I led the design of the project, developing the entire algorithm framework, verifying its feasibility.</strong>
            </p> -->
        </li>
    </ul>
</div>

<!-- 获取 GitHub Star 数量的 JavaScript -->
<script>
    async function fetchGitHubStars() {
        const repoOwner = 'yang-yang-o-o';  // 替换为 GitHub 用户名
        const repoName = 'torchchat';  // 替换为 GitHub 仓库名
        const apiUrl = `https://api.github.com/repos/${repoOwner}/${repoName}`;

        try {
            const response = await fetch(apiUrl);
            if (response.ok) {
                const data = await response.json();
                const starCount = data.stargazers_count;
                document.getElementById('star-count').textContent = `⭐ ${starCount} Stars`;
            } else {
                console.error('Failed to fetch star count:', response.status);
            }
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    // 页面加载完成后执行
    document.addEventListener('DOMContentLoaded', fetchGitHubStars);
</script>

