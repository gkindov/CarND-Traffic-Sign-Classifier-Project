#**Traffic Sign Recognition** 

##Writeup By Georgi E. Kindov

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

[image1]: ./writeup-images/train_set_vis.png "Visualization"
[image2]: ./new-images/2.jpg "Traffic Sign 1"
[image3]: ./new-images/7.jpg "Traffic Sign 2"
[image4]: ./new-images/9.jpg "Traffic Sign 3"
[image5]: ./new-images/10.jpg "Traffic Sign 4"
[image6]: ./new-images/12.jpg "Traffic Sign 5"
[image7]: ./new-images/13.jpg "Traffic Sign 6"
[image8]: ./new-images/14.jpg "Traffic Sign 7"
[image9]: ./new-images/20.jpg "Traffic Sign 8"
[image10]: ./new-images/25.jpg "Traffic Sign 9"
[image11]: ./new-images/28.jpg "Traffic Sign 10"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/gkindov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validaiton set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

The only technique I used was to shuffle the training images similar to the LeNet Lab example. Other preprocessing turn out to be not necessary to achieve the desired accuracy, but would be a good exercise to try gray-scaling or normalization to possibly get better results.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used training, validation, and testing set as they came in from the pickle file provided for this project.

A possible opportunity for improvement would be to join the two sets and randomly split them again to cross validate my model.

My training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the LeNetMod function of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					| RELU6 activation function						|
| Normalization			| Local Response Normalization					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					| RELU6 activation function						|
| Normalization			| Local Response Normalization					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 				    |
| Fully connected		| Input flattened 600, outputs 300				|
| Sigmoid				| Sigmoid activation 							|
| Fully connected		| Input 300, outputs 130        				|
| Sigmoid				| Sigmoid activation 							|
| Fully connected		| Input 130, outputs 43         				|															
I started as suggested with the LeNet-5 sample model. Since the accuracy out of the box was only about 87% I performed multiple adjustments to the model architecture to reach validation accuracy of 96%, as described in the next section.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper-parameters such as learning rate.

The code for training the model is located in the three cell below the section "Train, Validate and Test the Model" in the ipython notebook. 

To train the model, and improve the architecture of it I used a GPU machine on Azure cloud.
I started with the given LeNet-5 model and tweaking one thing at a time through trial and error I ended up making the following final changes (roughly in this order):
* increased the epochs to 32 - generally the more epochs the better (until accuracy peaks)
* increased the depth of the two convolution layers - since the original architecture was designed for a single channel grayscale image, I decided that giving mode depth to the convolution layers will aid better accuracy
* increased the width of the three fully connected layers - same idea as the previous - more weights to work with
* changed the activation functions from regular relu to relu6 for the ones after convolutions - looked at the list of the tensorflow activation functions library and played with some until a satisfactory was found
* changed the activation functions from regular relu to sigmoid for the ones after fully connected layers - same as the previous
* lowered the learning rate to 0.0005
* changed the stddev/sigma to 0.05
* introduced a normalization step after each convolution activation to prevent neurons from saturating - borrowed the idea from the AlexNet model - normalization is useful to prevent neurons from saturating when inputs may have varying scale, and to aid generalization

After the model was trained to satisfaction, it was saved using the tensorflow saver, which enables it to be loaded on other machines or later times without having to re-run the heavy training step.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located immediately after the "Train, Validate and Test the Model" sections.

My final model results were:
* validation set accuracy of 96% 
* test set accuracy of 95.6%

Looking at the validation accuracy at each epoch it looks like the architecture peaked at epoch 12 and started over-fitting and plateauing from there. A possible improvement could be a dropout layer to reduce over-fitting, possibly combined with even lower learning rate. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten color German traffic signs that I found on the web, and cropped to 32x32 size.

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

* The first image might be difficult to classify because it is slightly angled
* The second image might be difficult to classify because it is slightly angled
* The third image might be difficult to classify because there is a corner of another sign in the bottom left corner
* The fourth image might be difficult to classify because it low res and blurry
* The fifth image might be difficult to classify because it has busy background
* The sixth image might be difficult to classify because it is slightly angled
* The seventh image might be difficult to classify because the background seems to blend in a little
* The eight image might be difficult to classify because it has some background distractions
* The ninth image might be difficult to classify because the white inside part of the sign is yellow
* The tenth image might be difficult to classify because it has some distractions on the left side

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in below the section "Predict the Sign Type for Each Image" of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (50km/h)   						| 
| Speed limit (100km/h) | Speed limit (100km/h)							|
| No passing			| No passing									|
| No passing 3.5 tons	| No passing for vehicles over 3.5 metric tons	|
| Priority road			| Priority road      							|
| Yield			        | Yield                							|
| Stop			        | Stop                							|
| Dangerous curve right | Dangerous curve to the right      			|
| Road work 			| Road work      							    |
| Children crossing		| Children crossing      						|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.6%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located below section "Predict the Sign Type for Each Image" of the Ipython notebook.

The model seems quite sure for all ten images, with just three of them with lower than .99 soft max probability. The lowest was the road work sign with yellow instead of white inside which seems to explain the lower score. The top one soft max probabilities were, please refer to the notebook for the top 5.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         		    | Speed limit (50km/h)   						| 
| .98                   | Speed limit (100km/h)							|
| .98			        | No passing									|
| .99	                | No passing for vehicles over 3.5 metric tons	|
| .99       			| Priority road      							|
| .99			        | Yield                							|
| .99			        | Stop                							|
| .99                   | Dangerous curve to the right      			|
| .95        			| Road work      							    |
| .99           		| Children crossing      						|
