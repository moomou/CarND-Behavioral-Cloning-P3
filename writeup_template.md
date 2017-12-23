# **Behavioral Cloning**

## Writeup Template

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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

On a high level, the model uses repeating blocks 5x5 convolution (CNN) layers, 1x1 CNN layers, and 5x5 CNN layers with one batch normalization and ReLU after the 1x1 CNN layer.

The block is repeated twice and an attention based dense layer follows with dense layers.

For details on how this is derived as well as the final schematic, see the following sections.

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers between the last classify layers with probability 0.5 and the input images are downsized by half with guassian blur applied.

Additionally, the dataset is augmented with horizontal flip to double the dataset and left and right camera images utilized.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track at speed 9mph and 15mph.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as data augmentation via horizontal flipping.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adapt the model from traffic sign project.

From there, I experimented with filter size as well as the kernel size of the convolution layers. I also applied an attention based dense layer hypothesizing this will help guide the model to focus on the lane instead of background scenary. Ablation study of whether attention layer helped is left for future work.

I settled on kernel size 5 and an increasing filter size from 32 to 64 for convolution layer.

To combat the overfitting, I modified the model to incorporate dropout layers as well as applying data augmentation.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so I collected more training data dealing those specific section of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. See video.mp4 in the repo for a demonstration.

#### 2. Final Model Architecture

```
x (InputLayer)                  (None, 80, 160, 3)   0
__________________________________________________________________________________________________
cropping2d_1 (Cropping2D)       (None, 45, 160, 3)   0           x[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 45, 160, 3)   0           cropping2d_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 41, 156, 32)  2432        lambda_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 41, 156, 32)  1056        conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 41, 156, 32)  128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 41, 156, 32)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 37, 152, 32)  25632       activation_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 18, 76, 32)   0           conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 14, 72, 64)   51264       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 14, 72, 64)   4160        conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 14, 72, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 14, 72, 64)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 10, 68, 64)   102464      activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 5, 34, 64)    0           conv2d_6[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5, 34, 64)    4160        max_pooling2d_2[0][0]
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 5, 34, 64)    0           max_pooling2d_2[0][0]
                                                                 dense_1[0][0]
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 5, 34, 64)    0           multiply_1[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 10880)        0           lambda_2[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          5571072     flatten_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 512)          0           dense_2[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 512)          0           activation_3[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 256)          131328      dropout_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 256)          0           dense_3[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 256)          0           activation_4[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            257         dropout_2[0][0]
==================================================================================================
Total params: 5,894,209
Trainable params: 5,894,017
Non-trainable params: 192
__________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![ex1][./asset/ex1.jpg]
![ex2][./asset/ex2.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to center

![recovery][./asset/recovery.jpg]

Then I repeated this process on track one in reverse direction.

To augment the data sat, I also flipped images and angles. Here is an image that has then been flipped:

![flipped][./asset/recovery_flip.jpg]


After the collection process, I had the following data points:

    Total train samples:: 28668
    Total test samples:: 3186

I then preprocessed the data as suggested in the lectures with downsizing and guassia blur and finally normalization plus cropping as suggested in the lectures.

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 15  as evidenced by diverging validation loss vs training loss. I used an adam optimizer.
