# CIFAR-10-Image-Classifcation done with Pytorch

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:
![alt text](saved_images/CIFAR-10.png)

# The Results

Using this dataset, a **Fully-Connected output, 1 fully-connected hiden layer** model was tested.
With the best training accuracy being achieved **53.26\%**. 

Secondly, a **Convolutional Neural Network** with max pooling and fully connected output achieved
a training accuracy of **64.34\%**


