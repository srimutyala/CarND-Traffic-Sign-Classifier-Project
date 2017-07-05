#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./index.png "Data 1"
[image10]: ./images-small/1.jpg "Traffic Sign 1"
[image11]: ./images-small/2.jpg "Traffic Sign 2"
[image12]: ./images-small/3.jpg "Traffic Sign 3"
[image13]: ./images-small/4.jpg "Traffic Sign 4"
[image14]: ./images-small/5.jpg "Traffic Sign 5"
[image15]: ./images-small/6.jpg "Traffic Sign 6"

###Data Set Summary & Exploration

The dataset is divided into 3 portions, training, validation & test. We train the model on the training test, use the validation to check for error rate and adjust until satisfied. Finally, the model’s accuracy is tested against the test set which the model hasn’t seen before.

Some details regarding the dataset:
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (34799, 32, 32, 3)
Number of classes = 43

Example of a traffic sign image being used:

![alt text][image9]


Pre-Processing:
Several pre-processing techniques have been considered for this project and almost all of them have been tried and tested. A few of them are:
1.	Converting the images to different color-spaces (YUV,HSL, HSV, YCbCr)
2.	Using grayscale image as opposed to a 3-channel color image
3.	Contrast enhancement, histogram equalization (CLAHE), image sharpening
4.	Image normalization

In the end, I decided to keep the image in the RGB space with just image normalization being the only transformation done. 


###Design and Test a Model Architecture

MODEL:
I started out with lenet as a baseline and started experimenting with some additional layers to it. One significant addition was the Inception module. This raised the accuracy of the model. After a few trial and error runs involving different number of inception layers, pooling (max & avg) & fully connected layers, I arrived at my final model.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution w/RELU     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Convolution w/RELU					|		1x1 stride, valid padding, outputs 24x24x48										|
| Inception w/ Max Pooling		      	| outputs 12x12x96 				|
| Convolution					|		1x1 stride, valid padding, outputs 12x12x48									|
| Flatten		| output 6912        									|
| Fully Connected w/ RELU & Dropout				| outputs x120        									|
|	Fully Connected						|outputs x43												|
 
The model is trained for 10 epochs with batch size of 128. The learning rate is set to 0.0002. The final validation accuracy is 93.5% & the test accuracy is 93.1%.

(The excel file 'ExperimentalLog' conatins a few iterations of the model that were tested & discarded)

###Test a Model on New Images

A search for German sign images on the internet yielded a few signs that ranged from decent to bad quality. A few were handpicked(with varying degrees of quality) and resized to 32x32x3. The model was run on these images and the results are presented below.

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]

Images 1, 2, 3, & 5 were predicted accurately but images 4 & 6 were not. Based on a quick look at the images itself reveals that the accurately predicted ones have distinct features that are apparent to human eye (color, shape) wheres as the images that could not be determined accurately have colors that seem to blend into the background (sky).

Comparing selective images from the internet against a standard set( test data set in this case) might not be a very good indicative of the model's accuracy. While looking at different such non-standard images, it seems the model might not ve very well-adpated to scenes with perspective change, multiple signs in the image and any damage to the sign( dirt, paper stuck to the sign, etc). 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| Kee Right   									| 
| Priority Road     			| Priority Road 										|
| No Entry					| No Entry											|
| Turn right ahead	      		| Turn right ahead					 				|
| Road Work			| Road Work      							|
| Pedestrain & Bicycle Path			| Road Work      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This is worse than the test accuracy of 93.5%.


We used softmax & top_k to see the top 5 probabilities for each of those additional images. The probabilities and the corresponding labels for these are listed below. At a quick glance, we can see that the probabilities for the first 5 accurately predicted images are close to 1 (except for 1 which still had the highest probability for the correct label). 

Edit: Looking through the training dataset, the last image is not a part of 43 provided labels for the sign. Since there is no training for this particular sign, the model obviosuly failed to recognize it and tried to provide the best match.

[  1.00000000e+00   1.25504102e-24   2.73641724e-29   3.13950730e-30
   2.89225918e-31]
   
['Keep right', 'Turn left ahead', 'Go straight or right', 'Speed limit (60km/h)', 'Roundabout mandatory']

[  1.00000000e+00   2.22400761e-11   1.09336612e-13   1.35595863e-14
   6.95945817e-15]
   
['Priority road', 'Yield', 'Turn left ahead', 'End of no passing by vehicles over 3.5 metric tons', 'No vehicles']

[  1.00000000e+00   5.34872945e-11   3.96101641e-12   6.92277941e-13
   2.71545536e-13]
   
['No entry', 'Priority road', 'No passing', 'Speed limit (50km/h)', 'Speed limit (30km/h)']

[  5.83146513e-01   4.13416594e-01   2.19251378e-03   6.61697995e-04
   5.23580762e-04]
   
['Turn right ahead', 'Ahead only', 'Turn left ahead', 'Road work', 'Go straight or right']

[  1.00000000e+00   1.17906702e-13   2.50709338e-16   2.11312618e-16
   9.34069408e-19]
   
['Road work', 'Stop', 'No entry', 'Wild animals crossing', 'Bumpy road']

[  9.99984384e-01   1.46937218e-05   8.88643854e-07   9.18414500e-09
   5.17244303e-10]
   
['Roundabout mandatory', 'Keep right', 'Priority road', 'Go straight or right', 'Ahead only']




