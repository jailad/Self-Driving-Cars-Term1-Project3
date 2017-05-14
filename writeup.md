# Writeup for Self-Driving-Cars-Term1-Project3 - Behavioral Cloning

[//]: # (Image References)

[image1]: ./images/boxwhisker.png "boxwhisker.png"
[image2]: ./images/datasetdescription.png "datasetdescription.png"
[image3]: ./images/histograms.png "histograms.png"
[image4]: ./images/vis.png "vis.png"
[image5]: ./images/nvidia_cnn_architecture.png "nvidia_cnn_architecture.png"
[image6]: ./images/inceptionv3.png "inceptionv3.png"

* A submission by Jai Lad

# Table of contents

1. [Objective(s)](#objective)
2. [Key File(s)](#keyfiles)
3. [Other File(s)](#otherfiles)
4. [Detailed Description of Solution Approach](#da)
    1. [Pick five different CNN architectures for evaluation.](#da1)
    2. [Implementing the architectures.](#da2)
    3. [Quick end-to-end validation of pipeline.](#da3)
    4. [Evaluate different CNN architectures.](#da4)
    5. [Select a CNN architecture.](#da5)
    6. [Visualize the final CNN architecture.](#da6)
    7. [Data analysis and visualization.](#da7)
    8. [Data summary.](#da8)
    9. [Data balancing strategy.](#da8)
    10. [Data preprocessing.](#da9)
    11. [Data loading.](#da10)
    12. [Training, validation and testing.](#da11)
    13. [Off-center recovery practice.](#da12)
5. [Conclusion](#conc)

---

###Objective  <a name="objective"></a> : 

* Design, train and validate a model that predicts a steering angle from image data - model.h5 .
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
<BR><BR>

---

###Key File(s) <a name="keyfiles"></a> :

* readme.md - The accompanying Readme file, with setup details.
* model.ipynb - [Jupyter](http://jupyter.org/) notebook used to create and train the model.
* drive.py - Script to drive the car.
* model.h5 - A trained Keras model.
* writeup.md - A report writeup file
* video.mp4 - Video of the model.h5 driving successfully on track 1.
<BR><BR>
---

###Other File(s) <a name="otherfiles"></a> :

* Term1Project3BehaviorCloning.ipynb - the main Jupyter notebook which was used to experiment with various model architectures, and which was also used to perform data distribution analysis and balancing. This is the main 'parent' notebook which was used to complete the project, and from this, the model.ipynb was derived.

* SideExperiments.ipynb - another Jupyter notebook which was used to conduct small side experiments for this project.
<BR><BR>
---

###Detailed Description of Solution Approach <a name="da"></a> :

<BR>

#### - Pick five different CNN architectures for evaluation. <a name="da1"></a>
* The problem statement can be boiled down to an image recognition / regression task in which for a given image, we need to predict the ideal value of the steering angle. 
* [Convolutional neural network(s)](http://cs231n.github.io/convolutional-networks/) - CNN are the networks of choice for image recognition tasks, therefore I decided to use this overall architecture for this problem. In particular, in an image the nearby pixels are strongly correlated with each other, but the pixels farther from each other are not that significantly correlated with each other. This allows for sharing weights ( parameters ) between nearby pixels, and this is a property which CNNs are able to exploit well.
* Within CNNs we can have many different architecture(s), so I decided to select an initial set of five different CNNs for evaluation:

    1. Simple model. ( Term1Project3BehaviorCloning.ipynb / get_simple_model() )
    2. My [CNN Model](Self-Driving-Cars-Term1-Project2/Traffic_Sign_Classifier_dev.ipynb) from a previous project. ( Term1Project3BehaviorCloning.ipynb / get_my_model() )
    3. Modified [Nvidia](https://arxiv.org/pdf/1604.07316v1.pdf) model. ( Term1Project3BehaviorCloning.ipynb / get_nvidia_model() )
    4. A model architecture provided by [Comma.AI](https://github.com/commaai/research/blob/master/train_steering_model.py). ( Term1Project3BehaviorCloning.ipynb / get_commaai_model() )
    5. [Google Inception v3 Model](https://keras.io/applications/#inceptionv3) with Transfer Learning. ( Term1Project3BehaviorCloning.ipynb / get_inception_model() ). Additional information about Inception v3 is [here](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html).

<BR>

#### - Implementing the architectures <a name="da2"></a>
* I chose [Keras](https://keras.io/) with a TensorFlow backend.
* Reasons to do this are primarily because Keras supports constructing deep network(s) with much less code than TensorFlow.
* At the same time, with a Tensorflow back-end, it allows us to still train on GPUs which is one of the strengths of Tensorflow.
* It also provides sensible defaults for many of the hyper parameters. Hyper parameter initialization and tuning is a hard problem, and Keras provides a head start in that regard. 
* Gives excellent APIs to [inspect models and network hierarchies](https://keras.io/models/about-keras-models/).
* Supports several well-known [built-in models](https://keras.io/applications/) for Transfer Learning - e.g. Inception, VGG16, VGG19 etc.
* Also has good support to [visualize](https://keras.io/visualization/) the training process over multiple epochs.
* It will ( has ?) now be [officially integrated](https://github.com/fchollet/keras/issues/5050) into TensorFlow, and therefore has also has a vote of approval from the Tensorflow team.

<BR>

#### - Do quick end-to-end validation - training, validation, testing, driving. <a name="da3"></a>
* Created a simple CNN model architecture (Term1Project3BehaviorCloning.ipynb / get_simple_model() ), trained it over a small subset of the data, generated a model.h5 file, then launched the Simulator, and tried to drive the model with this initial model. This was done as a 'de-risking' / 'end-to-end' validation approach, so that I basically tested the complete chain of tasks from model training to model driving. This was done because I did not want to be in a situation where I had a reasonably trained model, which was not able to drive because of some quirks in the complete chain. 
* The car does not drive well at this stage, which is expected, but at least we have proven the end-to-end chain at this stage.

<BR>

#### - Evaluate different CNN architectures <a name="da4"></a>
* To rapidly evaluate the model(s), I did the initial evaluation with a few epochs.
* To perform this evaluation, I used the sample data provided as part of the starter project. The training data was shuffled for each epoch to ensure that the learning was reasonably independent of the ordering of data.
* I made sure that all the models were subjected to the same preprocessing. ( Term1Project3BehaviorCloning.ipynb / get_base_model() ) and the same amount of dropout ( const_dropout_probability ). 
* [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) is a process designed to reduce overfitting. However, when we are initially designing a model architecture, we want the model to show a good enough performance on the training data. In other words, at least when starting to train a model, overfitting is a better problem to have than underfitting because that implies that the model is not learning much about the training data. 
* To help the models learn faster initially, I set the dropout to zero.
* I used a training-to-validation split of 1:1 to ensure that the model(s) were able to perform equally or comparably well on training data and validation data. 
*  After the completion of training for each of the model(s), I noted the following parameters, for each of the above model(s):
    * Training accuracy.
    * Validation accuracy.
    * Training loss.
    * Validation loss.
    * Training time.
    * File Size of a trained model.
* Additionally, I saved the trained model as an H5 file ( with Keras checkpoints ), with the validation accuracy being saved as part of the filename, which allowed for a quick visual inspection the the accuracy of the model without having to access it via Keras. Also, using Checkpoint(s) helped me evaluate model(s) after each epoch, even before training for all the epoch(s) was complete. I also plotted the trend of validation loss over time, to check how this was progressing for a particular model and it ensure that it progressed successively. Finally, once the epochs were complete, I also tested the model, by using it for driving on the track. 

<BR>

#### - Select a CNN architecture for the project <a name="da5"></a>
* With the above specified evaluation parameter(s), I shortened the list of models under evaluation from five to two - the modified Nvidia model, and my model from a previous project. 
* I believe, that given sufficient training cycles, the CommaAI model, and the Inception V3 model would achieve or exceed the desired levels of accuracy.
* However, for my considerations, I was able to achieve the desired initial level(s) of performance with the modified Nvidia model, and with my model from a previous project, for much less memory ( 62.5 MB - model.h5 - Nvidia versus 500 MB+ - CommaAI model ) and in less time ( approx 400 seconds per epoch training time for Nvidia model, versus 1000 seconds per epoch for the CommaAI model. )
* The reason for the large size of the CommaAI model, is because it has a large number of [parameters](http://stackoverflow.com/questions/35792278/how-to-find-number-of-parameters-of-a-keras-model) to tune. The large number of parameters is a result of using 'Same' padding for convolutions ( versus using 'Valid' padding ), and also because of the lack of pooling layers in the architecture, which help in dimensionality reduction.
* I was also able to utilize a pre-trained Inception model, then re-purpose it for transfer learning, by removing it's top-most fully connected layer, then adding a small two layer feed-forward network at the end, and freezing all the layers, except the newly added feed forward network, and then training it on the evaluation data. However, after the small number of evaluation epochs, the Nvidia model and my model had a higher validation accuracy and lower validation loss on the data, so I decided to discard the Inception model. I believe that this result was because I had not normalized the inputs being passed into this model. When I started working with the Inception model, I was facing a compilation issue. Upon closer inspection I found that when I constructed the new feed forward network, I did not flatten the outputs of the Convolution layer, and this was leading to the issue. Once I added that layer, I was able to get past compilation issue. The diagram below is a schematic for the Inception V3 model.

![alt text][image6]


* Finally, between Nvidia model and my model, the Nvidia model had slightly better numbers for validation accuracy and validation loss than my model, at a slight cost of increased model size and training time. Because of this I decided to proceed with the Nvidia model for completing the actual project. I updated the Nvidia model to have a dropout layer at the very end, to avoid over fitting.
* Activation function = ReLU, to introduce non-linearity into the model.
* Padding strategy = Valid, to reduce the feature dimensions successively.
* Pooling strategy = Max Pooling, to reduce the feature dimensions successively.
* Loss / Cost function = Mean Squared Error, because this is a regression problem.
* Optimizer = [Adam](https://keras.io/optimizers/#adam), because it contains built-in support for learning rate decay, and is also one of the most [popular](https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106) optimizers currently. 


<BR>

#### - Visualize the final CNN architecture. <a name="da6"></a>

<BR>

![alt text][image5]

<BR>

#### - Data Analysis and Visualization <a name="da7"></a>


* The Data recorder produces a CSV file with the following columns, path to image from center camera, left camera, right camera, [ground truth](https://en.wikipedia.org/wiki/Ground_truth) values for steering angle, throttle, brake and speed.

* Initially, I wrote Python code to read the CSV file(s), and to generate basic statistics like mean, standard deviation, min and max for the steering angles.

* However, this appears to be a common enough problem, so I searched for a better solution. I subsequently discovered that Pandas is already optimized for task(s) such as these, and subsequently discarded my code in the favor of using Pandas, and converted this to a function which could take in path of a CSV file, and generate stats and visualization for that file. (Term1Project3BehaviorCloning.ipynb / generate_stats_and_visualization() )

![alt text][image2]

<BR>

* Generated [Box and Whisker plots](https://en.wikipedia.org/wiki/Box_plot) for the data sets.

![alt text][image1]

<BR>

* Generated Histograms for the data sets. 

![alt text][image3]

<BR>

* I visualized a few images from the CSV file. ( Term1Project3BehaviorCloning.ipynb / visualize_image_from() ). OpenCV reads images in BGR format, so before visualizing them via MatPlotLib, they had to be converted back to RGB format.

![alt text][image4]

<BR><BR>

#### - Data Summary <a name="da8"></a>

* From the above Steering angle Histogram, a few specific conclusions can be made:

    * We have a lot of data centered around steering angle of 0. This is intuitive because we ( purposely ) drive around center of the track.
    * The area under the histogram on the negative side is > area under the histogram on the positive side. This is a direct result of the shape of the track ( curving towards left > curving towards right ).
    * We have very less data for 'extreme angles'. That is, the availability of data reduces as we move farther away from 0.

<BR>

#### - Data balancing strategy <a name="da9"></a>

* The data summary above helps in preparing a data balancing strategy as follows:

    * We have concluded that we have a lot of data for the data range -0.1 < steering angle < 0.1. Therefore, it is safe to assume, that beyond a threshold, giving the model additional training for this data range will not result in any additional learning. An intuition behind this, is that say, you have a 1000-page book in which the content of every page is the same or very similar, then if you continue to read every single page of the book, you will not necessarily gain new knowledge. Furthermore, it also increases your reading time, because you would still be going over every single page. Thus, I initially chose to reject 70% of example(s) which met this range, and then evaluated the center driving performance. I was not very happy with the center driving performance with this rejection rate, so I reduced the rejection rate to 50%, and with threshold, I was getting good center driving results. A side benefit of doing this was that the number of training examples reduced significantly, and therefor the training time also reduced significantly.
    * Next, I drove the car in the opposite direction, so as to generate data in which additional positive steering angle data was captured.
    * Finally, in a few cases, I purposely drove off-center, and recorded the recovery process of driving to the center. I was careful to not record the part of the drive when I was driving off-center !
    * A good property of image data which makes it suitable for synthetic data generation, is that you can apply simple transforms on images, and 'generate' synthetic data which is equivalent to the original data, ( e.g. a rotated A, is still 'A' ) yet different from it in terms of representation. From a previous project, I also had convenience function(s) ready to generate synthetic data, if needed, via Greyscaling and random rotations of existing images.
    * With above data generated, I believed that I had a good balanced data set to work with and next commenced the training process.

<BR>

#### - Data preprocessing  <a name="da10"></a>
* Data Normalization
    * Data normalization is a process designed to help a network achieve convergence faster. Depending upon the type of problem, the type of normalization can vary. For [images](https://en.wikipedia.org/wiki/Normalization_(image_processing)), this can be performed by ensuring that the pixel intensities have zero mean, and unit variance. This was implemented as a Keras Lambda layer. This helps in reducing the number of epochs needed to train a neural network to an acceptable level of performance. 

* Region of interest selection
    * Within a single image, we can see that the area above the road ( horizon / sky ) might not necessarily be helpful to make decisions. Additionally, the lower part of the image contains the hood of the car which is again not particularly useful for making driving decisions. Therefore, these areas were rejected, with the help of another Keras Lambda Layer. A side benefit of this approach was that the size of images reduced as well, and therefore reduced the training time. The one crucial aspect to consider here is that convolution layers with Valid padding, and also Pooling layers lead to a reduction in the size of the image successively. Therefore, if we reduce the size of an image significantly, and if we have enough dimensionality reduction, then we might end up with an 'invalid' network, because the image size has already been reduced to near zero. I had anticipated this issue, so after I performed the region of interest selection, and ran into this issue, I re-adjusted the convolution and pooling layers to yield a valid architecture.

<BR><BR>

#### - Data Loading <a name="da11"></a>
* When I tried to load the images using paths from the CSV file(s), there was an extra space in some of the paths which led to failure, so those had to be stripped out.
* Other than this, when recording data, the image paths recorded were absolute, whereas in the reference data set, the image paths, the recorded paths were relative. So, this had to be accounted for.
* Finally, the sample data set had headers, whereas the data set generated while driving the car on the simulator did not have headers, which had to be [accounted for](https://docs.python.org/2/library/csv.html#csv.Sniffer) as well.
* With the above changes, I tried to load a sample of 100 images in-memory, and this was successful.
* However, the different data sets contain more than 15,000 images.
* As soon as I tried to load these images, I ran into memory issues.
* Image memory calculation = Image Dimensions (320,160) * Number of Channels (3) * Storage Format (Integer / Float) * Number of bytes for that format.
* In a non-normalized image, the pixel intensity is stored as an integer, whereas, upon normalization, it is stored as a float.
* Therefore, post-normalization, the same image would need more space in memory.
* Therefore, loading all images in memory at the same time, is not a scalable solution.
* The solution to this problem is to use a [Python generator approach](http://stackoverflow.com/questions/1756096/understanding-generators-in-python). To understand this concept, I did a quick PoC in which I compared returning first 'n' natural numbers with a traditional approach, and then with a generator approach. (Term1Project3BehaviorCloning.ipynb -> firstn_generator(), firstn_nongenerator() ) 
* Next, based on the above, I wrote an Image generator, which returned just the specific images needed for a particular batch of a training epoch at a particular time. ( Term1Project3BehaviorCloning.ipynb / load_images_generator()). So, say for example, I am training with a batch size of 100, then at one time only 100 images would be returned by this function, until it has iterated over the complete set.
* Subsequently, the image generator was used to define the training image generator, and the validation image generator.
* Finally, Keras provides several methods which have a postfix of '_generator' signifying that instead of taking raw data as inputs, they take generators for those data elements. One such function of interest, is '[fit_generator](https://keras.io/models/sequential/)'. The training and validation image generators obtained above, were passed into fit_generator, to perform training and validation. The batch size is a function of how much RAM you have for training, and for my case 100 was good enough. 

<BR><BR>

#### - Training and evaluation <a name="da12"></a>
* For the early evaluation, I had set dropout to 0, however for the actual training and evaluation, I set dropout value to 0.2, which meant that during the training phase, 20% of the activations would be randomly ignored. This was done to prevent overfitting.
* The initial training was performed for images captured from the central camera with the balanced data set described above. 
* Initial number of epochs were around 5.
* As explained previously, I used [checkpointing](https://keras.io/callbacks/#example-model-checkpoints) to generate models after each epoch of the training process, so that I could use those to view the driving performance of the model, even as training was progressing.
* At this stage, I also implemented [Early Stopping](https://keras.io/callbacks/#earlystopping), so that if the validation loss did not improve over a few epochs, we would stop the training process and re-examine the parameters / strategy.
* Even though the car drove properly at the center in most cases, there was a specific curve after the first bridge which was fairly sharp, and the car used to drive off the track at that curve. 
* Additionally, if the car moved away from the center, it's recovery characteristics back to center were less than ideal.

<BR><BR>

#### - Center driving and Off-center recovery practice. <a name="da13"></a>

* To resolve the above issues, I incorporated left camera data, and right camera data. Additionally, I generated a small offset value for these camera values as well ( positive for left camera, and negative for right camera). The intuition for this was that from the left camera, the left lane appeared to be closer ( relative to viewing the left lane from the center camera ), and we wanted to give the car a 'tendency' to increase the distance from the left lane when we got too close to it. Similarly, when we are looking at an image from the right camera, the right lane will appear to be close ( relative to right lane distance from center camera), and we want the car to increase the distance from the right lane. After experimentation and validation, I found that an offset value of 0.01 worked well in improving car's off-center recovery characteristics.
* I also did additional training by doing and recording recovery laps and training the model on this additional data. 
* These were the final numbers for the model -> 

| Result         	|     Value	        						| 
|:---------------------:|:---------------------------------------------:| 
| Loss         			| 0.0261   										| 
| Precision				| 161160.3707	 								|
| Recall				| 0.0094										|
| Validation Loss       | 0.0229   										| 
| Validation Precision  | 161134.3861	 								|
| Validation Recall		| 0.0161										|


<BR><BR>
---

###Conclusion <a name="conc"></a> :

* At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 
* The model was trained at 9mph, however, it is able to drive itself upto 25 mph.
* Result video is available within the Repo as video.mp4.
* Result video is also available via Youtube below :

[!['Behind the car' perspective](http://i.imgur.com/noZKpqd.jpg)](https://www.youtube.com/watch?v=6fGMyVC2dEs "Remixed version of the above video")

<BR><BR>
---