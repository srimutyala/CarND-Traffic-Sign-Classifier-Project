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

###Data Set Summary & Exploration

The dataset is divided into 3 portions, training, validation & test. We train the model on the training test, use the validation to check for error rate and adjust until satisfied. Finally, the model’s accuracy is tested against test set which the model hasn’t seen before.

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
I started out with lenet as a baseline and started experimenting with some additional layers to it. One significant addition was the Inception module. This raised the accuracy of the model. After a few trial and error runs involving different number of inception layers, pooling & fully connected layers, I arrived at my final model.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution w/RELU     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Convolution w/RELU					|		1x1 stride, valid padding, outputs 10x10x48										|
| Inception w/ Max Pooling		      	| outputs 12x12x128 				|
| Inception w/ Max Pooling		    | outputs 6x6x256      									|
| Convolution					|		1x1 stride, valid padding, outputs 6x6x96									|
| Flatten		| output 400        									|
| Fully Connected w/ RELU & Dropout				| outputs 6x6x120        									|
|	Fully Connected						|outputs x43												|
|						|												|
 
The model is trained for 30 epochs with batch size of 128. The learning rate is set to 0.0002.


My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?


###Test a Model on New Images

A search for German sign images on the internet yielded a few signs that ranged from decent to bad quality. A few were handpicked(with varying degrees of quality) and resized to 32x32x3. The model was run on these images are the results are presented below.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



