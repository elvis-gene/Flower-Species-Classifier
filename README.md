## Developing an image classifier with Deep Learning

In this project, An image classifier was built and trained (on a flower data set)  with **Pytorch** using a pre-trained deep neural network. The image classifier was trained to recognize different species of flowers (You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at). The project was firstly written in a _Jupyter Notebook_ then converted to file a command line  application.

### Technologies used:
* PyTorch
* Python
* Numpy
* Matplotlib
* GPU

### Training & Testing
A test accuracy of 82% was reached during training - using the train set.
A test accuracy of 84% was reach after training testing the classifier on the test set.

* #### Images

    * Before running the application on your computer, add a directory (named: **flowers**) to the project directory with 3 sub-directories (**train**, **test** and **valid**) which will need to be populated with images to be used to train.

### The Command Line Application
train.py trains a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.

##### Train a new network on a data set with train.py
The following basic usage will help you run the application with _default arguments_.
* Basic usage: python train.py
    * Prints out training loss, validation loss, and validation accuracy as the network trains

##### Predict image class with predict.py
* Basic usage: python predict.py
    * Return top KK most likely classes: python predict.py input checkpoint --top_k 3

***_The certificate I obtained after the completion of this projection at [Udacity](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) can be found [here](https://graduation.udacity.com/confirm/MEC3HUP)_***
