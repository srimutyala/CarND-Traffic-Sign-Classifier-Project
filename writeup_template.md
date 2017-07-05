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


We used softmax & top_k to see the top 5 probabilities for each of those additional images. The probabilities and the corresponding labels for these are listed below. At a quick glance, we can see that the probabilities for the first 5 accurately predicted images are close to 1 where as the last one(predicted inaccurately) has a probability of 0.6.

Edit: Looking through the training dataset, the last image is not a part of 43 provided labels for the sign. Since there is not training for this particular sign, the model obviosuly failed to recognize it and tried to provide the best match.

[  1.00000000e+00   1.72765922e-24   5.65642676e-29   1.65495611e-34
   9.46601623e-36]
['Keep right', 'Turn left ahead', 'Road work', 'Bumpy road', 'Dangerous curve to the right']

[  1.00000000e+00   2.38376904e-15   1.30654581e-16   3.35488575e-17
   3.21168788e-17]
['Priority road', 'End of all speed and passing limits', 'No vehicles', 'End of no passing by vehicles over 3.5 metric tons', 'No passing']

[  1.00000000e+00   3.46826623e-10   3.79998567e-11   7.76918113e-12
   7.68914619e-12]
['No entry', 'Priority road', 'Road work', 'No passing', 'Double curve']

[  7.13666499e-01   2.86146522e-01   1.76409230e-04   4.29482361e-06
   2.53876442e-06]
['Turn right ahead', 'Ahead only', 'Roundabout mandatory', 'Go straight or right', 'Turn left ahead']

[  1.00000000e+00   2.31675096e-12   4.21001463e-14   1.05129833e-15
   1.94352729e-16]
['Road work', 'Yield', 'Bicycles crossing', 'Speed limit (80km/h)', 'Go straight or right']

[  6.02692246e-01   2.81043440e-01   8.42467174e-02   3.20172124e-02
   3.22023681e-07]
['Keep right', 'End of no passing', 'General caution', 'Roundabout mandatory', 'Go straight or left']



