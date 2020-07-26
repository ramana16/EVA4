
<b><h1>Session 14 - Dataset For Monocular Depth Estimation and Segmentation</h1></b>

<b>Objective</b>

Create a custom dataset for monocular depth estimation and segmentation simultaneously.

Since we do not have access to a depth camera, we use a pre-trained depth model to generate the depth maps which will be used as the ground truth for our model.

<b>Dataset Samples</b>

Background:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Background.PNG"></img>

Foreground:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Foreground.PNG"></img>

Foreground Mask:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Foreground%20Mask.PNG"></img>

Foreground-Background:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Foreground-Background.jpg"></img>

Foreground-Background Mask:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Foreground%20Background%20Mask.jpg"></img>

Foreground-Background Depth:

<img src="https://github.com/ramana16/EVA4/blob/master/EVA4S14-15/Dense%20Depth.png"></img>


<b><h2>Dataset Creation</h2></b>

<b>Background (bg)</b>

<li>"scene" images. Like inside the office, etc.</li>
<li>100 images of streets were downloaded from the internet.</li>
<li>Each image was resized to 224 x 224</li>
<li>Number of images: 100</li>
<li>Image dimensions: (224, 224, 3)</li?
<li>Directory size: 2.5M</li>
<li>Mean: [0.5039, 0.5001, 0.4849]</li>
<li>Std: [0.2465, 0.2463, 0.2582]</li>
<br/>

<b>Foreground (fg)</b>

<li>Images of objects with transparent background
<li>100 images of footballers were downloaded from the internet.
<li>Using GIMP, the foreground was cutout. and the background was made transparent by adding an alpha layer.
<li>Each image was rescaled to keep height 105 and resizing width while maintaining aspect ratio.
<li>Number of images: 100
<li>Image dimensions: (105, width, 4)
<li>Directory size: 1.2M

<br/>
<b>Foreground Mask (fg_mask)</b>
<li>For every foreground its corresponding mask was created
<li>Using GIMP, the foreground was filled with white and the background was filled with black.
<li>Image was stored as a grayscale image.
<li>Each image was rescaled to keep height 105 and resizing width while maintaining aspect ratio.
<li>Number of images: 100
<li>Image dimensions: (105, width)
<li>Directory size: 404K
  
  <br/>
  
<b>Foreground Overlayed on Background (fg_bg)</b>
<li>For each background
    <li>Overlay each foreground randomly 20 times on the background
    <li>Flip the foreground and again overlay it randomly 20 times on the background
<li>Number of images: 100*100*2*20 = 400,000
<li>Image dimensions: (224, 224, 3)
<li>Directory size: 4.2G
<li>Mean: [0.5056, 0.4969, 0.4817]
<li>Std: [0.2486, 0.2490, 0.2604]
  
  <br/>
  
<b>Foreground Overlayed on Background Mask (fg_bg_mask)</b>
<li>For every foreground overlayed on background, its corresponding mask was created.
<li>The mask was created by pasting the foreground mask on a black image at the same position the foreground was overlayed.
<li>Image was stored as a grayscale image.
<li>Number of images: 400,000
<li>Image dimensions: (224, 224)
<li>Directory size: 1.6G
<li>Mean: [0.0454]
<li>Std: [0.2038]
  
 <br/>
 
<b>Foreground Overlayed on Background Depth Map (fg_bg_depth)</b>
<li>For every foreground overlayed on background, its corresponding depth map was generated.
<li>A pre-trained monocular depth estimation model DenseDepth was used to generate the depth maps.
<li>Image was stored as a grayscale image.
<li>Number of images: 400,000
<li>Image dimensions: (224, 224)
<li>Directory size: 1.6G
<li>Mean: [0.4334]
<li>Std: [0.2715]
  
 <b>Dataset Link</b>
 https://drive.google.com/file/d/1vkXFT3aADZx8xZCqFZ3mo2P3WmqofkmH/view?usp=sharing
 
<b>Resources</b>


Code to overlay foreground on background and corresponding masks:
https://github.com/ramana16/EVA4/blob/master/EVA4S15AP1.ipynb

Code to generate depth maps for foreground overlayed on background:
https://github.com/ramana16/EVA4/blob/master/EVA4S15DenseDepth.ipynb
