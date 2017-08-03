# Haze_noHaze
'''
deeplearning algorithm that predict if an image contain Haze or not.
'''

Step to reproduce the code:
___________________________________________
1) Data Preparation
- We start by creating the train and test  files to your our local machine. 
- Next, we run create_lmdb.py. don't forget to change the path to your train/test folders in create_lmdb.py file.
      usage : python create_lmdb.py

create_lmdb.py script does the following:

    (a)  Run histogram equalization on all training images. Histogram equalization is a technique for adjusting the contrast of images.
    (b)  Resize all training images to a 227x227 format.
    (c)  Divide the training data into 2 sets: One for training (5/6 of images) and the other for validation (1/6 of images). The training set is used to train the model, and the validation set is used to calculate the accuracy of the model.
    (d)  Store the training and validation in 2 LMDB databases. train_lmdb for training the model and validation_lmbd for model evaluation.
______________________________________________
2)  Generating the mean image of training data
  
  We execute the command below to generate the mean image of training data. We will substract the mean image from each input     image to ensure every feature pixel has zero mean:
    
      /home/ubuntu/caffe/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/deeplearning-cats-dogs-tutorial/input/train_lmdb /home/ubuntu/deeplearning-cats-dogs-tutorial/input/mean.binaryproto
    
______________________________________________
3)  Model Definition
After deciding on the CNN architecture, we need to define its parameters in a .prototxt train_val file. Caffe comes with a few popular CNN models such as Alexnet and GoogleNet. Here, we will use the bvlc_reference_caffenet model which is a replication of AlexNet with a few modifications.

We need to make the modifications below to the original bvlc_reference_caffenet prototxt file (here caffenet_train_val_hazy.prototxt):

    (a) Change the path for input data and mean image: Lines 24, 40 and 51.
    (b) Change the number of outputs from 1000 to 2: Line 373. The original bvlc_reference_caffenet was designed for a classification problem with 1000 classes.

We can print the model architecture by executing the command below :

    python /home/ubuntu/caffe/python/draw_net.py /home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt /home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1.png
 
 _________________________________________________
 4) Solver Definition
 
  
  
  
  

    
