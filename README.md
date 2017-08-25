# Udacity Self-Driving Car Engineer Nanodegree Program
---
## Behavioral Cloning

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/overfitting0.png "overfitting"
[image3]: ./examples/overfitting1.png "overfitting"
[image4]: ./examples/overfitting2.png "overfitting"
[image5]: ./examples/my_model.jpeg "My model"
[image6]: ./examples/recovery1.jpg "Recovery1"
[image7]: ./examples/recovery2.jpg "Recovery2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The file also contains the image preprocessing and augmenting methods used to fit the used architecture for our data.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modified version of [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used for end to end deep learning. 
The model is defined in the method my_model() (model.py lines 100-113)

This is the original model of Nvidia

![image1]


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


I've done more than 30 experiments with modifications in the model and data set to verify that the model won't be overfitting



1. At the beginning I used the original Nvidia architecture direct on the available training data provided by udacity without any modifications -> the result was a high overfitting over epochs and the performance was also not reliable
![image2]
2. I added dropout layers and the overfitting problem was somehow solved but the data was not enough to present all the critical cases that could happen. 
![image3]

3. I had to augment and preprocess (Drive clock wise, flip the images, crop, resize and convert to YUV) the data which lead to increasing the overfitting again with a higher MSE in both training and validation 
4. By increasing the percentage of the validation set splitted from the training set the overall validation loss has been reduced due to containing more cases for testing 
5. I reduced the number of epochs from 10 to 5 it helped in reducing the overfitting 
6. I had then several trials with the usage of the dropouts some of them worked in some speed with bias towards the left 
7. I've reached in many cases that there is no overfitting nor underfitting but the model doesn't handle all the cases in a stable way
8. I found that the dropout is not the best regularization in my problem and by analyzing the difference between it and the L2 regularization mentioned in [this paper](http://uksim.info/isms2016/CD/data/0665a174.pdf) the L2 regularization yields higher predictive accuracy than dropout in a small network since averaging learning model will enhance the overall performance when the number of sub-model is large and each of them must different from each other.
9. using the L2 regularization enabled me to eliminate the overfitting while giving an elegant performance in all cases 


![image4]

#### 3. Model parameter tuning

- The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123).
- The number of epochs have been tuned during the experiments many time to prevent overfitting and the final chosen number was 10 epochs

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, smooth drives around curves and one lap clock-wise 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy Details

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to be able to handle all the common and critical cases happens during the autonomous drive mode.

My first step was to use a convolution neural network model similar to the one used by Nvidia as it has been used for a similar operations which is to drive autonomously in any track so I chose it to be my start point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout layers which have been replaced afterwards with L2 regularization.

Then I've preprocessed the images to fit the model as the original image (160x320x3) and Nvidia image (66x200x3). So it is recommended to resize the input images for the model.

The final step was to run the simulator to see how well the car was driving around track one. At the beginning the vehicle drove well but was always tending to drive straight and in curves(specially sharp curves). After data preprocessing and data augmentation (which shall be mentioned afterwards) in addition to the L2 regularization all these issues have been fixed and the car had an elegant performance which is not just memorizing the training data but it could also predict smartly the best attitude in critical cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-113) 

Here is a visualization of the architecture (inspired by the architecture used by jeremy shannon)

![image5]





#### 3. Creation of the Training Set & Training Process

##### Data Collection
To capture good driving behavior: 


- I first recorded two laps on track one using center lane driving counter clockwise
- one lap clock wise to improve the generalization of the data set
- one lap with smooth slow driving on curves to capture much information there
- one lap of recovery driving from the left and right edges to the center. Here are two images of the start of recovery process

![image6] ![image7]


##### Left and Right camera Data:
I fed the model with the left and right camera images as if it is center images and corrected their steering angle with a factor of 0.25 (model.py lines 63-80). This also improved the model recovery and increased the amount of data set.

##### Data Preprocessing:

I decided to change in the images to be like the ones used by Nvidia in their model to be able to use the model perfectly so I had to do the following :

- cropping images 50 pixels from the top and 20 pixels from the bottom) so the image size converted from 160x320x3 to 90x320x3
- Convert images from BGR to YUV as cv2.imread() reads the images in BGR
- Re-size the image from 90x320x3 to 66x200x3

The last three operations have been done in model.py and in drive.py although in drive.py the second operation was to convert from RGB to YUV


##### Data Augmentation
To augment the data set, I also flipped images and angles thinking that this would improve the generalization as the car always pull to the left and to eliminate this bias I had to flip images to teach the model to turn right also if needed.


After the collection, preprocessing and augmentation processes I had 87960 number of data points.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.



####z 4. Conclusion

From the 30 experiments that I've done to reach the final output I've built up many theories:

- In this case the MSE of the training or validation will not give a solid measure to the real performance, it can only give an identification of overfitting which can be found while the performance is good 
- The data collection, preprocessing and augmenting is the most important factor in this case 
- The L2 regularization can be better than the dropout in small networks.

I got more than 5 working models with different parameters but I've chosen the one that I felt more comfortable with its performance and error graphs.

##### Upcoming Improvements
I didn't have a problem dealing with huge data set as I used FloydHub GPU and I saved the preprocessed data online. But I plan to use generators afterwards for making the model more reusable in lower resources.

