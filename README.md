# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Reducing Workplace Injuries in Singapore using Objection Detection Model

# Dillon Koh, DSIF-2 Singapore

## Problem Statement

Recently in 29th July 2021, a Straits Times article on Workplace inuries, fatalities can hinder business's recovery from Covid-19 pandemic was published. Manpower Minister Tan See Leng highlighted that employers and employees need to step up their workplace safety and health efforts as injuries and fatalities can hinder business recovery from the Covid-19 pandemic. He stressed that companies must continue their workplace safety and health efforts amid the pandemic, on top of the safe management measures to limit interactions at workplaces and prevent the spread of Covid-19 among workers.

In Workplace Safety and Health Report 2020 - National Statistics by Ministry of Manpower, it was mentioned that the top three causes of workpace minor injuries were:

​	(i) Slips, Trips & Falls

​	(ii) Machinery Incidents; and

​	(iii) Struck by Moving Objects.

These collectively accounted for 55% (5,993 cases) of the total number of workplace minor injuries in year 2020.

Hence, I would like to make use of Deep Learning to reduce the number of workplace injuries in Singapore. 

The overall end goal of my projet is to reduce the number of workplace injuries by using Object Detection Model to detect if any workers are not wearing PPE (Helmet and Vest) properly inside a PPE zone and Sounding off an alarm to alert the workers on the infringement using real time monitoring by cameras. However, the scope of my GA capstone will only cover the Object Detection Model aspect.

**<u>Target Audience</u>** 

Various Stakeholders such as Worksite Contractors / Companies, Ministry of Manpower and etc

**<u>Datasets of Images</u>** 

* Kaggle
* Google Images

## Image Classification using EfficientNet
Before diving deep into Object Detection itself, I have decided to use CNN to do image classification of whether an image contains personnel who did not wear the safety PPE properly (not wearing helmet or/and safety vests) inside a PPE zone. The reasons for doing this is to apply my newly learned knowledge of CNN, leveraging on transfer learning and also evaluate if image classification is sufficient at this point.

There are two main benefits to using transfer learning:
1. Can leverage an existing neural network architecture proven to work on problems similar to our own.
2. Can leverage a working neural network architecture which has **already learned** patterns on similar data to our own. This often results in achieving great results with less custom data.

I will be doing the following - 
* Preparing 3 different sets of images where positive images are images which personnel worn the PPE properly and negative images are images which personnel worn the PPE wrongly (infringement of safety regulations). There will not be imbalanced data as I have sufficient images for the 2 different classes.

> - Training Images (70 positive and 70 negative)

>- Validation Images (220 positive and 220 negative)
>- Test Images (100 positive and 100 negative)

- Data augmentation built right into the model

- [`EfficientNetB0`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) architecture from `tf.keras.applications` as our base model

![](C:\Users\dillo\OneDrive\Desktop\EfficientNet.png)

- Sigmoid activation as the output layer
- Binary crossentropy as the loss function since we're dealing with only 2 classes (positive or negative)
- The Adam optimizer with the default settings
- Fitting for 20 full passes on the training data 
- Predicting with the trained Model and illustrating several test images
- Predicting with the trained Model and evaluating the ambiguous images

### Summary of Results and Evaluation

**<u>Results from training the model</u>**

After compiling the model and training the model for 20 epochs, I obtained a loss of 0.2566, val_loss of 0.2209, accuracy of 0.9571 and val_accuracy of 0.9531. The model is performing relatively well and this is actually the power of transfer learning. In the later sections of the notebook, I will actually be using this model to predict on the test images and some ambiguous images. 

**<u>Results from test images</u>**

Subsequently, I used my trained model to make predictions with 200 test images (100 positive and 100 negative) and achieved the following -

- **Test accuracy score is 0.955**
- **Test Recall score is 0.960**
- **Test Precision score is 0.950**

Based on the above scores, the train model also seem to perform well with unseen images (from the Test datasets). However, these test images are images which are quite straightforward - everyone in the positive image will be wearing both the helmet and safety vest properly and similarly, everyone in the negative do not have helmet and safety vest. I thought that it would be good to give my model a good challenge - ambiguous images where some but not all personnels are wearing the PPE properly and see the results. 

**<u>Results from ambiguous images</u>**

The model seem to perform poorly when required to predict images which are considered ambiguous. In these images, personnel are wearing only helmet or safety vest and in some images, one person will be wearing full PPE but the other person will be wearing only half PPE which ultimately should predict as a negative class. Out of 10 ambiguous images, 9 images were predicted as positive with about 5 (which is 50%) of the images having a probability of greater than 0.75 which translate to predicting it as positive.

Based on this results, the model seem to predict positive when a person is wearing either a helmet or safety vest. This is understandable as the training images are trained using negative images where people are not wearing helmets and safety vests at all. 

In conclusion, it seems like to tackle the problem statement I have, image classification is insufficient. I have to use object detection to identify the 3 different objects and at the same time locate the objects in the images. This will be covered in subsequent sections.

## Object Detection using EfficientDet-Lite0 Model

In this second notebook of my Capstone Project, I will train a custom object detection model using Tensorflow Lite Model Maker Library (TF API) leveraging on transfer learning with EfficientDet-Lite0 model. EfficientDet-Lite[0-4] are a family of mobile/IoT-friendly object detection models derived from the EfficientDet architecture.

Here is the performance of each EfficientDet-Lite models compared to each others.

| Model architecture | Size(MB) | Latency(ms) | Average Precision |
| ------------------ | -------- | ----------- | ----------------- |
| EfficientDet-Lite0 | 4.4      | 37          | 25.69%            |
| EfficientDet-Lite1 | 5.8      | 49          | 30.55%            |
| EfficientDet-Lite2 | 7.2      | 69          | 33.97%            |
| EfficientDet-Lite3 | 11.4     | 116         | 37.70%            |
| EfficientDet-Lite4 | 19.9     | 260         | 41.96%            |

\* Size of the integer quantized models.

** Latency measured on Pixel 4 using 4 threads on CPU.

*** Average Precision is the mAP (mean Average Precision) on the COCO 2017 validation dataset.

The EfficientDet architect employs EfficientNet as the backbone network, BiFPN as the feature network and the architecture can be shown in the figure below -

<img src="C:\Users\dillo\OneDrive\Desktop\EfficientDet.PNG" style="zoom:100%;" />

### Preparing the images

Before training my object detection model, I have to prepared the images and annotations to train my object detection model. As mentioned in the above, I have images downloaded from both Kaggle and Google. They consists of images of people, safety vests and helmets. I will then use LabelImg which is a grahpical image annotation tool which is written in Python and ues Qt for its graphical interface to draw bounding boxes on safety vests, person and helmets in all my images. The annotation will then be saved as XML files in PASCAL VOC format which is compatible to the Tensorflow Lite Model Maker Library.

An example of the annotated images is shown in the figure below.

![](C:\Users\dillo\OneDrive\Desktop\Labelimg pic.PNG)

Upon completion of the annotation of all images, i then proceed to split my images an respective annotations into 3 different groups -

- Training Images - 70%
- Validation Images - 20%
- Test Images - 10%

The training images will be used to train the object detection model to recognize helmets, safety vests and persons. The validation images are images that the model didnt see during the training process and will be used in hyperparameter tuning and stop the training. Finally, the test images will be used to evaluate the performance of the model as new data it has never seen before.

Also, for several images, I have also used an image augmentation tool to perform a specific set of augmentation operations on each file that it finds. Some operations include horizontal and vertical flips, rotation, translation, zoom/stretch and blurring the images.

### Approach to training the Object Detection Model

Before training the custom Object Detection Model with entire dataset (total of 1,213 images), I will a 10% of my images to ensure that our smaller modelling experiments are working. This is a common practice in machine learning and deep learning: get a model working on a small amount of data before scaling it up to a larger amount of data.

![](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/06-ml-serial-experimentation.png)

**Machine learning practitioners are serial experimenters. Start small, get a model working, see if your experiments work then gradually scale them up to where you want to go.** Source: https://colab.research.google.com/drive/1vBR85tR5_yS88ZhKf2XfzKKKKAoqhLou?usp=sharing#scrollTo=UTWetPM7AWfY

After confirming that the model with 10% images is completed with no problems. I will then train my model with the entire datasets.

I trained a model with 50 epochs and 25 epochs and evaluated the results of their performance. Subsequently, I also export the models to a tflite model and evaluate the performance after exporting. The main reason is there will be several factors that can affect the model accuracy when exporting to TFLite:

* Quantization helps shrinking the model size by 4 times at the expense of some accuracy drop.

* The original TensorFlow model uses per-class non-max supression (NMS) for post-processing, while the TFLite model uses global NMS that's much faster but less accurate. Keras outputs maximum 100 detections while tflite outputs maximum 25 detections.

Therefore I'll have to evaluate the exported TFLite model and compare its accuracy with the original TensorFlow model.

### Evaluation Metrics and Summary of Results

After training the model, I will be evaluating the test data of both models which are trainined with 50 and 25 epochs considering the mean average precision (mAP) of the test datasets. The library computes the mAP which is the detection evaluation metrics use by Common Objects in Context (COCO). In fact, there are 15 metrics that used for characterizing the performance of an object detector on COCO and be found in the following link - https://cocodataset.org/#detection-eval.

However, for my project I will only consider the following 3 mAPs of the test instead of the other metrics:

* mAP (at IoU=50:.05:.95) - Uses 10 IoU thresholds of 0.50 to 0.95 with step of 0.05 and average the AP across the different thresholds [More Stringent]

* mAP (at IoU=50) - Consider the AP at IoU of 0.50 [Traditional Method]

* mAP (at IoU=0.75) Consider the AP at IoU of 0.75 [In between the above 2]

The reason is mAP is a popular metric in measuring the accuracy of object detection model. 

In addition, I will actually do a manual evaluation of my chosen model by doing some error analysis running through the predicted bounding boxes on all my test images and evaluating the model performance.  For the selection of the model, I will purely use the 3 mAPs described above

**<u>Model results</u>**

The following table summarizes the mAPs scores -

| Model                                      | mAP            | mAP Score |
| ------------------------------------------ | -------------- | --------- |
| Model with 50 epochs (Before Quantization) | IoU=50:.05:.95 | 0.613     |
|                                            | IoU=50         | 0.914     |
|                                            | IoU=0.75       | 0.713     |
| Model with 50 epochs (After Quantization)  | IoU=50:.05:.95 | 0.582     |
|                                            | IoU=50         | 0.880     |
|                                            | IoU=0.75       | 0.674     |
| Model with 25 epochs (Before Quantization) | IoU=50:.05:.95 | 0.593     |
|                                            | IoU=50         | 0.905     |
|                                            | IoU=0.75       | 0.679     |
| Model with 25 epochs (After Quantization)  | IoU=50:.05:.95 | 0.566     |
|                                            | IoU=50         | 0.889     |
|                                            | IoU=0.75       | 0.630     |

Based on the scores, the model with 50 epochs perform better and hence I will use the tflite model as my chosen object detection model.

### Manual Evaluation of Test Images using Trained Object Detection Model

After selecting the tflite model, I will be evaluating the bounding boxes predicted by the trained object detection model based on my judgement and methodology to have an intuition on ways to improve the performance of my model. After conducting some research, I have decided on my evaluation methodology which is based on how the mean average precision (mAP) score is derived.

First, I will go through all 100 images and classify the performance of the model on each object and bounding boxes based on the following definition:

* True Positive (TP) - Classify the object detection if IoU > 0.5

* False Positive (FP) - Classify the object detection if IoU < 0.5 or Duplicated Bounding Box or 1 Bounding Box bounding 2 objects

* False Negative (FN) - When a ground truth is present in the image and model failed to detect the object, or Wrong classification predicted

* True Negative (TN) - TN is every part of the image where we did not predict an object. This metrics is not useful for object detection, hence we ignore TN.

Based on the above definitions and methodology, the Precision and Recall scores of 0.973 and 0.782 are achieved respectively. The reason for a lower recall score is because of 102 False Negatives were observed. The confusion matrix is shown below:

![](C:\Users\dillo\OneDrive\Desktop\confusion matrix capstone.PNG)

After going through 100 test images, I have also come up with a summary of the error analysis and I have also provided my reasons behind the errors committed by the the trained model -

* Resolution of images are quite bad in general which could have resulted in the model to not be able to predict several objects

* I have trained my model using 800+ images which may not be sufficient and training with more images would likely lea to my model to perform better 

* False Negative Analysis - Fails to detect the following objects: Helmet - 40%; Person - 35%; Safety Vest - 25%

> - Possible Reasons - Insufficient images for various angles, positions of the objects. Example, a person wearing at helmet and tilting his head at 45 degrees angle, certain part of the object is blocked 

* False Positive Analysis - 1 Bounding Box tend to group 2 identical objects together

> - Possible Reasons - For some images, the Bounding Box was drawn to overlap several objects due to the object being close to each other and some object overlap other objects too.

Considering my evaluation results and the mAP score obtained, I believed that the performance of my object detection model is quite good and require minor fine tuning before it can be deployed to actual worksite for trial.

### Video Processing using Trained Object Detection Model

At this current stage, I have achieved 50% of overall end goal which I would like to achieve from this Capstone Project. The second part of the goal is Sounding off an alarm to alert the workers on the infringement using real time monitoring by cameras. As deploying the model on real time camera feed is not possible as I do not have access to a camera system. I will do a mock up of this by running my object detection model on a video instead and I have uploaded the video into my git repo too.

Based on the predicted bounding boxes drawn by my model, I believed additional work is still required to improve the performance of the model. This will be covered in the last section - Future Works.

## Future Works

In this section, I will be elaborating the key points and aspects to achieve my overall end goal (which goes beyond the scope of my Capstone Project) and tackle the problem statement. It is very important to have an accurate object detection model. Hence the following steps are my proposal to deploy it in operation:

**Understanding each client's needs and requirements**

- The accuracy of the model depends on several factors and is unique to different clients and hence the training of objection detection model can further be fine tuned based on the client's requirements and situation

  > - Resolution and quality of the camera feed - To train the model with images of the similar resolution of the camera feed
  > - Location and type of the cameras - to better understand the position, angle and etc of the objects.
  > - Helmets, safety vests and Uniforms of the company such as covered overall which also function as safety vest. To train the model with these images

- Discuss with the client on how would they like the model to be deployed and the platform to view the results i.e. web browser or mobile app to view the camera feeds with the object detection model running on the camera feeds

- Providing a complete solution by integrating the object detection model, camera system and sensory system (to identify the PPE zone), alarm system with a control system (using a simple controller if possible) to monitor and trigger an alarm when any infringement is detected.

**Improving the performance of the Object Detection Model**

- Training the model with more images of the client's requirements and situation
- Using image augmentation tool to augment the images such as rotation, flipping, introducing noise to provide more of such images when training the model to get a more robust model
- Consistent Drawing of the Bounding Box when annotating the images

**Proof of Concept/Trial**

A small scale deployment of the end product can be put on trial and the results can be evaluated after a month. Once the client is satisified and confident of the model, we can then proceed to deploy it on a wider scale or full operation depending on the client's decision.

# References

- https://www.mom.gov.sg/-/media/mom/documents/safety-health/reports-stats/wsh-national-statistics/wsh-national-stats-2020.pdf

- https://www.straitstimes.com/singapore/jobs/workplace-injuries-fatalities-can-hinder-business-recovery-from-covid-19-pandemic-tan

- https://www.kaggle.com/johnsyin97/hardhat-and-safety-vest-image-for-object-detection

- https://amaarora.github.io/2021/01/11/efficientdet.html

- https://amaarora.github.io/2020/08/13/efficientnet.html

- https://blog.paperspace.com/mean-average-precision/

- https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52

- https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py

- https://dzone.com/articles/how-to-use-google-colaboratory-for-video-processin

- https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

*  https://www.tensorflow.org/lite/tutorials/model_maker_object_detection

- https://arxiv.org/abs/1911.09070

- https://github.com/tzutalin/labelImg

- https://cocodataset.org/#detection-eval

- https://www.tensorflow.org/lite/tutorials/model_maker_object_detection

- https://www.tensorflow.org/lite/performance/post_training_quantization
