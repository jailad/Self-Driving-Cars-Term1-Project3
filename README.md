# Read me for Self-Driving-Cars-Term1-Project3 - Behavioral Cloning

[//]: # (Image References)

[image1]: ./images/boxwhisker.png "boxwhisker.png"
[image2]: ./images/datasetdescription.png "datasetdescription.png"
[image3]: ./images/histograms.png "histograms.png"
[image4]: ./images/vis.png "vis.png"
[image5]: ./images/nvidia_cnn_architecture.png "nvidia_cnn_architecture.png"

* A submission by Jai Lad

# Table of contents

1. [Objective(s)](#objective)
2. [Result Videos](#resultvideos)
    1. [Drivers' perspective](#resultvideos1)
    2. [Behind the car perspective](#resultvideos2)
    3. [Remix of the above!](#resultvideos3)
3. [Pre-Requisites / Setup / Dependencies](#prereq)
4. [A note on using GPUs for training](#gpu)
5. [Packages used to build the pipeline ( model.ipynb )](#packages)
6. [Key File(s)](#keyfiles)
7. [Other File(s)](#otherfiles)
8. [High Level Approach](#hla)
    1. [Pick five different CNN architectures for evaluation.](#hla2)
    2. [Implement the architectures.](#hla3)
    3. [Quick end-to-end validation of pipeline.](#hla1)
    4. [Evaluate different CNN architectures.](#hla4)
    5. [Select a CNN architecture.](#hla5)
    6. [Data analysis and visualization.](#hla6)
    7. [Data summary.](#hla7)
    8. [Data balancing strategy.](#hla8)
    9. [Data preprocessing.](#hla9)
    10. [Data loading.](#hla10)
    11. [Training, validation and testing.](#hla11)
    12. [Off-center recovery practice.](#hla12)
9. [Conclusion](#conc)
10. [Future Work and Learning](#fw)

---

## Objective  <a name="objective"></a> : 

* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior. 
* Design, train and validate a model that predicts a steering angle from image data - [model.h5](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/model.h5) .
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report - [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/writeup.md)

---

## Result videos <a name="resultvideos"></a> :

* The following videos are as captured during autonomous driving by my trained model. ( model.h5 )

### - Drivers' perspective <a name="resultvideos1"></a> :

[![Drivers' perspective](http://i.imgur.com/zJLOHPq.jpg)](https://www.youtube.com/watch?v=yzjdvTPG86Y "Drivers' perspective")

<BR><BR>

### - A behind the car perspective <a name="resultvideos2"></a> :

[!['Behind the car' perspective](http://i.imgur.com/YutMA4t.jpg)](https://www.youtube.com/watch?v=PDhiD3CyBcM "Behind the car perspective")

<BR><BR>

### - A mandatory Remixed version of the above video ! <a name="resultvideos3"></a> :
 
[!['Behind the car' perspective](http://i.imgur.com/noZKpqd.jpg)](https://www.youtube.com/watch?v=6fGMyVC2dEs "Remixed version of the above video")

<BR><BR>

---

## Pre-Requisites / Setup / Dependencies <a name="prereq"></a> : 

* [Term1 - starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)

    * I followed the instructions for Docker image ( Mac ) because I preferred to have the same code easily deployable to my local workstation ( with CPU ) versus GPU ( Amazon ), and Docker allows for that flexibility. This was also done to gain more familiarity with the Docker environment, and also because the setup was fairly straightforward.

    * Other than the above, I also setup the starter kit as the MiniConda environment which allowed me to easily interact with the generated model from the Command line. This also gave an opportunity to practise setup with the [Anaconda](https://conda.io/miniconda.html) environment.
<BR><BR>
---

## A note on using GPUs for training <a name="gpu"></a> : 

* With regards to [Amazon P2 GPU](https://aws.amazon.com/ec2/instance-types/p2/) instances, I was able to get good results with my local workstation because of which I did not setup the Amazon P2 instances for this project, though I have used it in the past for model training. There are quite a pre-built Amazon AMIs which contain TensorFlow, Keras and related libraries installed which can be used for experimentation on Amazon. [This](https://github.com/ritchieng/tensorflow-aws-ami) is an example of such an AMI. [Here](http://course.fast.ai/lessons/aws.html) is another useful resource on how to setup AWS P2 GPU instances. 
<BR><BR>
---

## Packages used to build the pipeline ( model.ipynb ) <a name="packages"></a> : 

* [Keras](https://keras.io/) - 1.2.1  ( with [Tensorflow](https://www.tensorflow.org/) backend )
* OpenCV - 3.1.0
* Numpy - 1.12.1
* [Pandas](http://pandas.pydata.org/) - 0.19.2
* MatPlotLib - 2.0.0
* SciKitLearn - 0.18.1
* Python 3.6 - resource, csv, time, datetime
<BR><BR>
---

## Key File(s) <a name="keyfiles"></a> :

* [model.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/model.ipynb) - [Jupyter](http://jupyter.org/) notebook used to create and train the model.
* [drive.py](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/drive.py) - Script to drive the car.
* [model.h5](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/model.h5) - A trained Keras model.
* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/writeup.md) - A report writeup file
* [video.mp4](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/video.mp4) - Video of the model.h5 driving successfully on track 1.
<BR><BR>
---

## Other File(s) <a name="otherfiles"></a> :

* [Term1Project3BehaviorCloning.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/Term1Project3BehaviorCloning.ipynb) - the main Jupyter notebook which was used to experiment with various model architectures, and which was also used to perform data distribution analysis and balancing. This is the main 'parent' notebook which was used to complete the project, and from this, the model.ipynb was derived.

* [SideExperiments.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project3/blob/master/SideExperiments.ipynb) - another Jupyter notebook which was used to conduct small side experiments for this project.
<BR><BR>
---

## High level overview of approach <a name="hla"></a> :

* For each of the steps below, the details are provided within the writeup.md file.

<BR>

### - Pick five different CNN architectures for evaluation. <a name="hla2"></a>

* I selected five initial architectures for implementing this problem.

<BR>

### - Implementing the architectures <a name="hla3"></a>

* The implementation platform chosen was Keras ( w/ Tensorflow backend ).

<BR>

### - Do quick end-to-end validation - training, validation, testing, driving. <a name="hla1"></a>

* This was done from a risk mitigation perspective, to prove out the end-to-end pipeline up front.

<BR>

### - Evaluate different CNN architectures <a name="hla4"></a>
*  The above selected models were competetively evaluated against each other.
*  Evaluation parameter(s):
    * Training accuracy.
    * Validation accuracy.
    * Training loss.
    * Validation loss.
    * Training time.
    * File Size of a trained model.

<BR>

### - Select a CNN architecture for the project <a name="hla5"></a>

* In this step, the final CNN architecture was selected for this project.

<BR>

### - Data Analysis and Visualization <a name="hla6"></a>

* In this step, descriptive statistics for sample data was generated, and the data set was visualized.
* Generated [Box and Whisker plots](https://en.wikipedia.org/wiki/Box_plot) for the data sets.
* Generated Histograms for the data sets. 
* I visualized a few images from the CSV file. 

<BR><BR>

### - Data Summary <a name="hla7"></a>

* From the above analysis, a summary was prepared, which was the foundation for the next step, which is coming up with a strategy for data balancing.

<BR>

### - Data balancing strategy <a name="hla8"></a>

* In this step, the data balancing strategy was finalized. Over represented data was rejected, where as under represented data was additionally captured or synthesized.

<BR>

### - Data preprocessing  <a name="hla9"></a>

* Data Normalization
* Region of interest selection

<BR><BR>

### - Data Loading <a name="hla10"></a>

* In this step, the challenges of loading the data were resolved with a generator pattern.

<BR><BR>

### - Training and evaluation <a name="hla11"></a>

* In this step, the model training and evaluation was performed.

<BR><BR>

### - Center driving and Off-center recovery practice. <a name="hla12"></a>

* In this step, the model was trained specifically for off-center recovery.


<BR><BR>
---

## Conclusion <a name="conc"></a> :

* At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 
* The model was trained at 9mph, however, it is able to drive itself upto 25 mph.

<BR><BR>
---

## Future Work and Learning <a name="fw"></a> :

* Ensuring that the model can drive at the top speed of 30 mph. 

* Ensuring that the model is able to drive well, on the much more challenging track 2.

* Experimenting with 'off-track' recovery. That is, when a model has actually left the road, is it able to use the Camera to recover back to the road.

* Extending this project by doing multi-class predictions, for adaptive throttle control implementation. The current model uses PID control to stay as close as possible to a fixed speed. However, by extending the model, we should be able to implement adaptive speed control, in which for a given scene, in addition to predicting the steering angle, we also predict the throttle value. For example, on a straight road, we should be able to zoom at the max speed, whereas on a curve, we can reduce the throttle to an appropriate value.

* Generate activation maps of different Keras layers for a given image. This would be helpful in seeing, 'how' the Keras layer sees a given image, i.e. what road features stand out for the network and help it in making the decisions.

* Currently, I have used a modified Nvidia architecture to drive the car. However, as an optimization exercise, I can try to shave off a few layer(s) from this architecture, and then evaluate it, until I can come up with a smaller architecture which drives 'well enough' on the track. The advantage of this would be smaller training time, and smaller memory footprint of a saved model.

* Given a pre-trained network I need to learn how to insert a pre-processing Lambda layer at the very start of the network.

---




