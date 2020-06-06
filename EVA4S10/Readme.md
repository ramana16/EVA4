Assignment Objective - 


Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
Please make sure that your test_transforms are simple and only using ToTensor and Normalize
Implement GradCam function as a module. 
Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
Target Accuracy is 87%
Submit answers to S9-Assignment-Solution. 

Quiz Objective - 

Make a DNN such that:

it's first block uses following code:
import datetime from datetime
print("Current Date/Time: ", datetime.now())
uses the Modules you have written
calls following DNN from a file called QuizDNN.py:
x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)
Uses ReLU and BN wherever applicable
Uses CIFAR10 as the dataset
Your target is 75% in less than 40 Epochs

