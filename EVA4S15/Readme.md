<h1>Session 15 - Monocular Depth Estimation and Segmentation</h1>
<br>

<b>Objective</b> <br/>
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

<b>Dataset</b><br/>
A custom dataset will be used to train this model, which consists of:

<li>100 background images
<li>400k foreground overlayed on background images
<li>400k masks for the foreground overlayed on background images
<li>400k depth maps for the foreground overlayed on background images
  
<b>Dataset Samples</b><br/>

Background: 
<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/bg_strip.png"></img>
<br/>
Foreground-Background: 

<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/bg_strip.png"></img>
<br/>
Foreground-Background Mask: 

<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/bg_strip.png"></img>
<br/>
Foreground-Background Depth: 

<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/bg_strip.png"></img>
<br/>

<b>Notations</b>
<li>Background image: <b>bg</b>
<li>Foregroung overlayed on background: <b>fg_bg</b>
<li>Mask for fg_bg: <b>fg_bg_mask</b>
<li>Depth map for fg_bg: <b>fg_bg_depth</b>
<li>Mask prediction:<b> mask_pred</b>
<li>Depth map prediction: <b>depth_pred</b>
  
<b>Model</b>
<br/>
The inputs to the model are <b>bg</b> and <b>fg_bg</b>.

<li>bg : [3 x 224 x 224]
<li>fg_bg: [3 x 224 x 224]
  <br>
The outputs of the model are mask_pred and depth_pred.

<li>mask_pred: [1 x 224 x 224]
<li>depth_pred: [1 x 224 x 224]
Model definition file: https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/depth_and_mask_dnn.py

<b>Architecture</b>
