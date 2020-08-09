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
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/fg_bg_strip.png"></img>
<br/>
Foreground-Background Mask: 

<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/fg_bg_mask_strip.png"></img>
<br/>
Foreground-Background Depth: 

<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/fg_bg_depth_strip.png"></img>
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
The outputs of the model are <b>mask_pred</b> and <b>depth_pred</b>.

<li>mask_pred: [1 x 224 x 224]
<li>depth_pred: [1 x 224 x 224]
<br/>
Model definition file: https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/depth_and_mask_dnn.py

<b>Architecture</b>
<br/>

The model follows an encoder-decoder architecture. It consists of a common encoder and two decoders, for mask prediction and depth map prediction respectively.

<li>The encoder uses ResNet blocks to extract the visual features.
<li>The decoder uses skip connections from the encoder and transpose convolutions to upscale the features and construct the mask and depth maps.
<br/>
<img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/dnn_architecture.png">
  <br>
 <b> Parameters Count</b>
  <br>
  <pre><code>--------------------------------------------------
Total params: 3,165,170
Trainable params: 3,165,170
Non-trainable params: 0
--------------------------------------------------
Input size (MB): 1.15
Forward/backward pass size (MB): 31581162963630.00
Params size (MB): 12.07
Estimated Total Size (MB): 31581162963643.22
--------------------------------------------------
</code></pre>
  
  <h3><a id="user-content-parameters-and-hyperparameters" class="anchor" aria-hidden="true" href="#parameters-and-hyperparameters"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Parameters and Hyperparameters</h3>
  
  <ul>
<li>Optimizer: Adam</li>
<li>Scheduler: StepLR</li>
<li>Batch Size:
<ul>
<li>64 for <code>64x64</code> and <code>128x128</code></li>
<li>16 for <code>224x224</code></li>
</ul>
</li>
<li>Dropout: 0.2</li>
<li>L2 decay: 0.0001</li>
</ul>

<h3><a id="user-content-image-augmentation" class="anchor" aria-hidden="true" href="#image-augmentation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Image Augmentation</h3>

<ul>
<li><strong>Resize</strong>:
<ul>
<li>Downscale the images to be able to train for lower dimensions first.</li>
<li>Applied on <strong>bg</strong>, <strong>fg_bg</strong>, <strong>fg_bg_mask</strong> and <strong>fg_bg_depth</strong>.</li>
</ul>
</li>
<li><strong>RGBShift</strong> &amp; <strong>HueSaturationValue</strong>:
<ul>
<li>Used to reduce the dependency on image colours for prediction.</li>
<li>One of these was applied randomly to <strong>bg</strong> and <strong>fg_bg</strong> images.</li>
</ul>
</li>
<li><strong>GaussNoise</strong>:
<ul>
<li>Gaussian noise was applied randomly to <strong>bg</strong> and <strong>fg_bg</strong> images.</li>
</ul>
</li>
<li><strong>Horizontal &amp; Vertical Flip</strong>:
<ul>
<li>Images were flipped randomly, both horizontally and vertically</li>
<li>Applied on <strong>bg</strong>, <strong>fg_bg</strong>, <strong>fg_bg_mask</strong> and <strong>fg_bg_depth</strong>.</li>
<li>Here, the same flip operations were applied on all the 4 images to maintain the orientation.</li>
</ul>
</li>
<li><strong>RandomRotate</strong>:
<ul>
<li>Images were randomly rotated within (-15,15) degrees.</li>
<li>Applied on <strong>bg</strong>, <strong>fg_bg</strong>, <strong>fg_bg_mask</strong> and <strong>fg_bg_depth</strong>.</li>
<li>Here, all the 4 images were rotated in the same direction and angle to maintain the orientation.</li>
</ul>
</li>
<li><strong>CoarseDropout</strong>:
<ul>
<li>Used to force the network to extract more features by cutting out some parts of the image</li>
<li>Applied randomly to <strong>bg</strong> and <strong>fg_bg</strong> images.</li>
</ul>
</li>
</ul>
