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

<p>Image augmentation file: <a href="https://github.com/ramana16/EVA4/tree/master/EVA4S15/data/data_transforms.py">https://github.com/ramana16/EVA4/tree/master/EVA4S15/data/data_transforms.py</a></p>

<h3><a id="user-content-loss-function" class="anchor" aria-hidden="true" href="#loss-function"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Loss Function</h3>

<p>The loss function used is a weighted average of:</p>

<ul>
<li><strong>L1 loss</strong>: Handles the per pixel differences.</li>
<li><strong>Structural Similarity Index (SSIM)</strong>: Handles luminance, contrast and structural differences.</li>
<li><strong>Edge Gradients</strong>:  This computes the edges and tries to match the edges of the target and output.</li>
</ul>

<p>The overall loss was a summation of loss for mask and depth.</p>

<pre><code>loss = (w_ssim * l_ssim) + (w_depth * l_depth) + (w_edge * l_edge)
loss = loss_mask + loss_depth
</code></pre>

<p>We can get results by just using <strong>L1 loss</strong> too, but using <strong>SSIM</strong> and <strong>Edge gradients</strong> helps in converging faster.</p>

<p><strong>Huber loss</strong> can also be used but I observed that the predictions are sharper when we use <strong>L1 loss</strong> instead of huber.</p>

<p><strong>BCE loss</strong> could construct the structure of the prediction but was not able to get a proper constrast for the mask and sharpness for mask and depth images.</p>

<p>Loss function file: <a href="https://github.com/ramana16/EVA4/tree/master/EVA4S15/loss.py">https://github.com/ramana16/EVA4/tree/master/EVA4S15/loss.py</a></p>

<h3><a id="user-content-accuracy-metrics" class="anchor" aria-hidden="true" href="#accuracy-metrics"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Accuracy Metrics</h3>

<p>We need some metrics to help us understand how the model is performing and to be able to compare two models. The notion of accuracy is different here because we have to compare two images.</p>

<h4><a id="user-content-root-mean-squared-error-rms" class="anchor" aria-hidden="true" href="#root-mean-squared-error-rms"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Root Mean Squared Error (RMS)</h4>

<ul>
<li>RMS is based on the per pixel differences.</li>
<li>This measure is in the context of the images being absolutely the same.</li>
<li>The lower the RMS error, the better the predictions</li>
</ul>

<p>Calculation:</p>

<pre><code>rmse = torch.sqrt(torch.nn.MSELoss()(gt, pred))
</code></pre>

<h4><a id="user-content-t--125" class="anchor" aria-hidden="true" href="#t--125"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>t &lt; 1.25</h4>

<ul>
<li>The idea here is that, the pixel values need not be asolutely the same.</li>
<li>We take the ratio of each pixel and verify if it is within a scale of 1.25</li>
<li>This measure is in the context of the images being relatively the same.</li>
<li>The higher the <code>t&lt;1.25</code> value, the better the predictions</li>
<li>Similarly calculate <code>t&lt;1.25^2</code>  and <code>t&lt;1.25^3</code></li>
</ul>

<p>While calculating <code>t&lt;1.25</code>, we want the ratio of pixels to be within a threshold i.e 1.25. But if the value of pixel is less than 0.1 then even though the pixel values are close the ratio scale changes.
For ex, 0.00001 and 0.000001 are very close and we want them to contribute positively for our accuracy but the ratio is 10 which reduces the accuracy. So we clamp the tensors to 0.1 and 1.</p>

<p>Calculation:</p>

<pre><code>gt = torch.clamp(gt, min=0.1, max=1)
pred = torch.clamp(pred, min=0.1, max=1)

thresh = torch.max((gt / pred), (pred / gt))

a1 = (thresh &lt; 1.25   ).float().mean()
a2 = (thresh &lt; 1.25 ** 2).float().mean()
a3 = (thresh &lt; 1.25 ** 3).float().mean()
</code></pre>

<p>Testing or validation file: <a href=https://github.com/ramana16/EVA4/tree/master/EVA4S15/test.py">https://github.com/ramana16/EVA4/tree/master/EVA4S15/test.py</a></p>
  
  <h3><a id="user-content-training-and-validation" class="anchor" aria-hidden="true" href="#training-and-validation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Training and Validation</h3>
  
  <p>The model was first trained on smaller resolutions of <code>64x64</code> first. Then trained on <code>128x128</code> and finally on <code>224x224</code>for faster convergence.</p>
  
  <p>Since there are storage and compute restrictions on colab, I was not able to use large batch sizes for higher resolutions and this in turn was increasing the time taken per epoch.
To handle this, I was saving the predictions and model after a chunk of batches to be able to monitor the progress while running for a small number of epochs. Using the saved model, I could again load it and continue training the model.</p>

<p>For <code>224x224</code> each epoch was taking ~2hrs to train. So I was able to train it for 3 epochs at a stretch, save the model and resume training later.</p>

<h3><a id="user-content-results" class="anchor" aria-hidden="true" href="#results"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Results</h3>

<h4><a id="user-content-64x64" class="anchor" aria-hidden="true" href="#64x64"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>64x64</h4>

<p>Dimensions:</p>

<pre><code>bg         : [3, 64, 64]
fg_bg      : [3, 64, 64]
fg_bg_mask : [1, 64, 64]
fg_bg_depth: [1, 64, 64]
mask_pred  : [1, 64, 64]
depth_pred : [1, 64, 64]
</code></pre>

<p>Validation Metrics:</p>

<pre><code>Test set: Average loss: 0.1857, Average MaskLoss: 0.0411, Average DepthLoss: 0.1445
  
Metric:  t&lt;1.25,   t&lt;1.25^2   t&lt;1.25^3,   rms
Mask  :  0.9808,   0.9866,    0.9900,    0.0604
Depth :  0.7569,   0.9244,    0.9743,    0.0909
Avg   :  0.8688,   0.9555,    0.9822,    0.0757
</code></pre>

<p>Saved Model Link: <a href="https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im64_l0.1863_rms0.0760_t0.8684.pth">https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im64_l0.1863_rms0.0760_t0.8684.pth</a></p>

<p>Visualization:</p>

<p><a target="_blank" rel="noopener noreferrer" href="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_64.png"><img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_64.png" style="max-width:100%;"></a></p>

<h4><a id="user-content-128x128" class="anchor" aria-hidden="true" href="#128x128"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>128x128</h4>

<p>Dimensions:</p>

<pre><code>bg         : [3, 128, 128]
fg_bg      : [3, 128, 128]
fg_bg_mask : [1, 128, 128]
fg_bg_depth: [1, 128, 128]
mask_pred  : [1, 128, 128]
depth_pred : [1, 128, 128]
</code></pre>

<p>Validation Metrics:</p>

<pre><code>Test set: Average loss: 0.1474, Average MaskLoss: 0.0253, Average DepthLoss: 0.1221

Metric:  t&lt;1.25,   t&lt;1.25^2,  t&lt;1.25^3,   rms
Mask  :  0.9891,   0.9925,    0.9947,    0.0409
Depth :  0.7558,   0.9205,    0.9722,    0.0926
Avg   :  0.8725,   0.9565,    0.9834,    0.0667
</code></pre>

<p>Saved Model Link: <a href="https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im128_l0.1473_rms0.0666_t0.8726.pth">https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im128_l0.1473_rms0.0666_t0.8726.pth</a></p>

<p>Visualization:</p>

<p><a target="_blank" rel="noopener noreferrer" href="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_128.png"><img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_128.png" style="max-width:100%;"></a></p>

<h4><a id="user-content-224x224" class="anchor" aria-hidden="true" href="#224x224"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>224x224</h4>

<p>Dimensions:</p>

<pre><code>bg         : [3, 224, 224]
fg_bg      : [3, 224, 224]
fg_bg_mask : [1, 224, 224]
fg_bg_depth: [1, 224, 224]
mask_pred  : [1, 224, 224]
depth_pred : [1, 224, 224]
</code></pre>

<p>Validation Metrics:</p>

<pre><code>Test set: Average loss: 0.1446, Average MaskLoss: 0.0231, Average Depthloss: 0.1214

Metric:  t&lt;1.25,   t&lt;1.25^2,  t&lt;1.25^3,   rms
Mask  :  0.9923,   0.9946,    0.9961,    0.0350
Depth :  0.7280,   0.9028,    0.9623,    0.1015
Avg   :  0.8601,   0.9487,    0.9792,    0.0682
</code></pre>

<p>Saved Model Link: <a href="https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im224_l0.1444_rms0.0682_t0.8593.pth">https://github.com/ramana16/EVA4/tree/master/EVA4S15/models/saved_models/im224_l0.1444_rms0.0682_t0.8593.pth</a></p>

<p>Visualization:</p>

<p><a target="_blank" rel="noopener noreferrer" href="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_224.png"><img src="https://raw.githubusercontent.com/ramana16/EVA4/master/EVA4S15/images/viz_224.png" style="max-width:100%;"></a></p>






